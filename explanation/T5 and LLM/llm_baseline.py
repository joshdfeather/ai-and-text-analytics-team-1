import json
import requests
from tqdm import tqdm

# ==========================================
# 1. set up
# ==========================================
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

# ==========================================
# 3. Baseline function
# ==========================================
def extract_decision(text):
    t = text.lower()
    if 'final decision: yes' in t: return 'yes'
    if 'final decision: no' in t: return 'no'
    if 'final decision: maybe' in t: return 'maybe'
    return 'unknown'

def run_baseline(question):
    prompt = f"""[Question]: {question}
provide a concise answer in no more than 100 words.
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
# 4. answering
# ==========================================
results = []
skipped = []
print(f"\nStart Baseline answering，total {len(test_pmids)} records...\n")

for pmid in tqdm(test_pmids):
    if pmid not in ori_data:
        skipped.append(pmid)
        continue

    content     = ori_data[pmid]
    question    = content['QUESTION']
    long_answer = content.get('LONG_ANSWER', '')

    llm_output  = run_baseline(question)

    results.append({
        "id":                pmid,
        "model_explanation": llm_output,
        "long_answer":       long_answer
    })

if skipped:
    print(f"\n⚠️  {len(skipped)} PMID in ori_pqal.json not found，skip: {skipped}")

# ==========================================
# 5. save
# ==========================================
output_path = "data/baseline_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 50)
print(f"finish！success {len(results)} ，skip {len(skipped)} ")
print(f"Result saved: {output_path}")
print("=" * 50)
