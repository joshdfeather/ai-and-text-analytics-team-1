import argparse
import importlib.util
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_n_sentences(text: str, n: int = 2) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()


def build_context_from_contexts(item: Dict) -> str:
    """Use CONTEXTS only as model input to avoid leaking LONG_ANSWER."""
    contexts = item.get("CONTEXTS") or []
    if isinstance(contexts, list):
        return " ".join(str(x).strip() for x in contexts if str(x).strip()).strip()
    return str(contexts).strip()


def build_t5_input(question: str, context: str, context_char_limit: int) -> str:
    c = (context or "")[:context_char_limit]
    return (
        "Generate a concise medical explanation.\n"
        "Given the question and context, explain the answer in 2-3 sentences.\n"
        f"Question: {question}\n"
        f"Context: {c}"
    )


class Seq2SeqDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: Dict[str, List[int]], pad_token_id: int):
        self.encodings = encodings
        self.labels = labels
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.labels["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        y = torch.tensor(self.labels["input_ids"][idx], dtype=torch.long)
        y[y == self.pad_token_id] = -100
        item["labels"] = y
        return item


def train_t5_explainer(
    model,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        avg = loss_sum / max(len(dataloader), 1)
        print(f"[t5-train] epoch={ep} avg_loss={avg:.4f}")


def predict_t5_explanations(
    model,
    tokenizer,
    inputs: List[str],
    device: torch.device,
    batch_size: int,
    max_input_len: int,
) -> List[str]:
    model.eval()
    outputs = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=128, do_sample=False)
        texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])
    return outputs


def generate_with_ollama(
    prompt: str,
    model_name: str,
    base_url: str,
    timeout_s: int = 120,
    retries: int = 3,
) -> str:
    body = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    payload = json.dumps(body).encode("utf-8")
    url = f"{base_url.rstrip('/')}/api/generate"

    last_err = ""
    for _ in range(retries):
        req = urllib.request.Request(
            url=url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip()
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0)
    return f"OLLAMA_ERROR: {last_err}"


def build_llm_prompt(question: str, context: str, context_char_limit: int) -> str:
    c = (context or "")[:context_char_limit]
    return (
        "You are a biomedical assistant.\n"
        "Based only on the context, provide a concise explanation (2-3 sentences).\n"
        "Do not output JSON.\n\n"
        f"Question: {question}\n"
        f"Context: {c}\n"
        "Explanation:"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run 500-id test_ground_truth explanations with T5 and LLM."
    )
    parser.add_argument("--ground_truth_path", type=str, default="./test_ground_truth.json")
    parser.add_argument("--data_path", type=str, default="./ori_pqal.json")
    parser.add_argument("--t5_model_name", type=str, default="t5-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--t5_context_char_limit", type=int, default=2500)
    parser.add_argument("--llm_model", type=str, default="llama3.1:8b")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm_context_char_limit", type=int, default=1600)
    parser.add_argument("--t5_output_jsonl", type=str, default="./test_ground_truth_t5_explanations.jsonl")
    parser.add_argument("--llm_output_jsonl", type=str, default="./test_ground_truth_llm_explanations.jsonl")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    t5_pipeline = load_module_from_path("t5_pipeline", os.path.join(base_dir, "t5_pipeline.py"))
    t5_pipeline.set_seed(args.seed)

    with open(args.ground_truth_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    test_ids = list(gt_map.keys())
    if len(test_ids) != 500:
        raise ValueError(f"Expected 500 ids in test_ground_truth, got {len(test_ids)}")

    missing = [qid for qid in test_ids if qid not in all_data]
    if missing:
        raise ValueError(f"{len(missing)} ids not found in ori_pqal.json. Example: {missing[:5]}")

    # Train split uses only non-test IDs to avoid data leakage.
    test_id_set = set(test_ids)
    train_ids = [qid for qid in all_data.keys() if qid not in test_id_set]
    overlap = test_id_set.intersection(set(train_ids))
    if overlap:
        raise ValueError(f"Leakage detected: train/test overlap size={len(overlap)}")

    train_rows = []
    for qid in train_ids:
        item = all_data[qid]
        long_answer = (item.get("LONG_ANSWER") or "").strip()
        context = build_context_from_contexts(item)
        target_expl = first_n_sentences(long_answer if long_answer else context, n=2)
        train_rows.append(
            {
                "question": item.get("QUESTION", ""),
                "context": context,
                "target_explanation": target_expl,
            }
        )

    # Prepare 500 test rows.
    test_rows = []
    for qid in test_ids:
        item = all_data[qid]
        long_answer = (item.get("LONG_ANSWER") or "").strip()
        context = build_context_from_contexts(item)
        test_rows.append(
            {
                "id": qid,
                "question": item.get("QUESTION", ""),
                "context": context,
                "long_answer": long_answer if long_answer else context,
            }
        )

    # Guardrail: input context should not be identical to LONG_ANSWER.
    suspicious = 0
    for row in test_rows:
        source_item = all_data[row["id"]]
        la = (source_item.get("LONG_ANSWER") or "").strip()
        if la and row["context"].strip() == la:
            suspicious += 1
    if suspicious:
        raise ValueError(
            f"Leakage guard failed: {suspicious} test rows use LONG_ANSWER as context."
        )

    # ===== T5 explanation =====
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model_name)

    train_inputs = [
        build_t5_input(r["question"], r["context"], args.t5_context_char_limit)
        for r in train_rows
    ]
    train_targets = [r["target_explanation"] for r in train_rows]

    x_enc = tokenizer(train_inputs, truncation=True, padding=True, max_length=args.max_input_len)
    y_enc = tokenizer(text_target=train_targets, truncation=True, padding=True, max_length=192)
    ds = Seq2SeqDataset(x_enc, y_enc, tokenizer.pad_token_id)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_t5_explainer(model, dl, device=device, epochs=args.epochs, learning_rate=args.learning_rate)

    t5_inputs = [
        build_t5_input(r["question"], r["context"], args.t5_context_char_limit)
        for r in test_rows
    ]
    t5_explanations = predict_t5_explanations(
        model=model,
        tokenizer=tokenizer,
        inputs=t5_inputs,
        device=device,
        batch_size=args.eval_batch_size,
        max_input_len=args.max_input_len,
    )

    with open(args.t5_output_jsonl, "w", encoding="utf-8") as f:
        for r, exp in zip(test_rows, t5_explanations):
            out = {
                "id": r["id"],
                "model_explanation": exp,
                "long_answer": r["long_answer"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ===== LLM explanation (Llama via Ollama) =====
    with open(args.llm_output_jsonl, "w", encoding="utf-8") as f:
        for idx, r in enumerate(test_rows, start=1):
            prompt = build_llm_prompt(r["question"], r["context"], args.llm_context_char_limit)
            exp = generate_with_ollama(
                prompt=prompt,
                model_name=args.llm_model,
                base_url=args.ollama_url,
            )
            out = {
                "id": r["id"],
                "model_explanation": exp,
                "long_answer": r["long_answer"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if idx % 25 == 0:
                print(f"[llm] generated {idx}/{len(test_rows)}")

    print(
        json.dumps(
            {
                "test_ids": len(test_rows),
                "t5_output_jsonl": args.t5_output_jsonl,
                "llm_output_jsonl": args.llm_output_jsonl,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

