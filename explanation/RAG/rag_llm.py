import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# 1. Set up
# ==========================================
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
OLLAMA_MODEL = "llama3.2"

# ==========================================
# 2. load
# ==========================================
print("loading...")

with open("data/ori_pqal.json", "r") as f:
    ori_data = json.load(f)

with open("data/test_ground_truth.json", "r") as f:
    ground_truth = json.load(f)

test_pmids = list(ground_truth.keys())
print(f"testset: {len(test_pmids)} records")

kb_pmids = list(ori_data.keys())
print(f"knowledge base: {len(kb_pmids)} records")

# ==========================================
# 3. vectorize
# ==========================================
print("\n Building a vector index ...")
kb_texts = [" ".join(ori_data[p]['CONTEXTS']) for p in kb_pmids]
kb_embeddings = embed_model.encode(kb_texts, show_progress_bar=True, batch_size=64)

dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(kb_embeddings).astype('float32'))

# ==========================================
# 4. RAG
# ==========================================
def run_rag(question):
    # question only
    query_vec = embed_model.encode([question])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k=3)

    retrieved = ""
    for i, idx in enumerate(indices[0]):
        retrieved += f"Evidence {i+1}: {kb_texts[idx]}\n\n"

    prompt = f"""You are a medical research expert.

[Retrieved Evidence]:
{retrieved}

[Question]: {question}

Based on the retrieved evidence, provide a concise answer in no more than 100 words.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 5. answer
# ==========================================
results = []
skipped = []
print(f"\n Start RAG, total {len(test_pmids)} ...\n")

for pmid in tqdm(test_pmids):
    if pmid not in ori_data:
        skipped.append(pmid)
        continue

    content     = ori_data[pmid]
    question    = content['QUESTION']
    long_answer = content.get('LONG_ANSWER', '')

    llm_output  = run_rag(question)

    results.append({
        "id":                pmid,
        "model_explanation": llm_output,
        "long_answer":       long_answer
    })

if skipped:
    print(f"\n⚠️  {len(skipped)} PMID in ori_pqal.json not find，skipped: {skipped}")

# ==========================================
# 6. Save
# ==========================================
output_path = "data/rag_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 50)
print(f"Finish! Success {len(results)}，skip {len(skipped)}")
print(f"Results saved: {output_path}")
print("=" * 50)
