import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

datasets_to_check = {
    "og_golden_dataset": "evals/og_golden_dataset.json",
    "golden_dataset": "evals/golden_dataset.json",
    "hard_retrieval_22": "evals/data/hard_retrieval_22.json",
    "guardrails_benchmark_v3": "evals/data/guardrails_benchmark_v3.json",
    "guardrails_benchmark_v2": "evals/data/guardrails_benchmark_v2.json",
    "guardrails_eval_22": "evals/data/guardrails_eval_22.json",
    "retrieval_golden_seed_v1": "evals/data/retrieval_golden_seed_v1.json",
}

for name, path in datasets_to_check.items():
    if not os.path.exists(path):
        print(f"[-] {name}: NOT FOUND at {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = 0
    keys = []
    sample = None
    if isinstance(data, list):
        count = len(data)
        if count > 0:
            sample = data[0]
            keys = list(sample.keys()) if isinstance(sample, dict) else []
    elif isinstance(data, dict):
        if "rag_samples" in data:
            count = len(data["rag_samples"])
            if count > 0:
                sample = data["rag_samples"][0]
                keys = list(sample.keys())
        elif "records" in data:
            count = len(data["records"])
            if count > 0:
                sample = data["records"][0]
                keys = list(sample.keys())
        elif "cases" in data:
            count = len(data["cases"])
            if count > 0:
                sample = data["cases"][0]
                keys = list(sample.keys())
def normalize_q(q):
    import string
    s = q.lower()
    for p in string.punctuation:
        s = s.replace(p, " ")
    return " ".join(s.split())

def token_jaccard(q1, q2):
    t1 = set(normalize_q(q1).split())
    t2 = set(normalize_q(q2).split())
    return len(t1 & t2) / len(t1 | t2) if (t1 and t2) else 0.0

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    gd15 = json.load(f)["rag_samples"]
with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
    hr22 = json.load(f)
with open("evals/data/retrieval_golden_seed_v1.json", "r", encoding="utf-8") as f:
    seed50 = json.load(f)["records"]

print("\n" + "=" * 90)
print("CROSS-DATASET COMPARISON WITH 50-RECORD SEED (retrieval_golden_seed_v1.json)")
print("=" * 90)

# Compare GD15 vs Seed50
print(f"\n--- Comparing Original Golden Dataset (15 Qs) vs Seed50 (50 Qs) ---")
exact_matches_15 = []
near_matches_15 = []
for g in gd15:
    g_q = g["question"]
    matched = False
    for s in seed50:
        s_q = s["query"]
        if normalize_q(g_q) == normalize_q(s_q):
            exact_matches_15.append((g["id"], s["id"], g_q, s_q))
            matched = True
            break
        elif token_jaccard(g_q, s_q) >= 0.70:
            near_matches_15.append((g["id"], s["id"], g_q, s_q, round(token_jaccard(g_q, s_q), 3)))
            matched = True
            break

print(f"Exact Matches: {len(exact_matches_15)} of 15")
for gid, sid, gq, sq in exact_matches_15:
    print(f"  [Q{gid} -> {sid}] Exact: \"{gq}\"")

print(f"Near Matches (Jaccard >= 0.70): {len(near_matches_15)}")
for gid, sid, gq, sq, jacc in near_matches_15:
    print(f"  [Q{gid} -> {sid}] Jaccard={jacc}: \"{gq}\" <-> \"{sq}\"")

# Compare HR22 vs Seed50
print(f"\n--- Comparing Hard Retrieval Benchmark (22 Qs) vs Seed50 (50 Qs) ---")
exact_matches_22 = []
near_matches_22 = []
for h in hr22:
    h_q = h["question"]
    for s in seed50:
        s_q = s["query"]
        if normalize_q(h_q) == normalize_q(s_q):
            exact_matches_22.append((h["id"], s["id"], h_q, s_q))
            break
        elif token_jaccard(h_q, s_q) >= 0.70:
            near_matches_22.append((h["id"], s["id"], h_q, s_q, round(token_jaccard(h_q, s_q), 3)))
            break

print(f"Exact Matches: {len(exact_matches_22)} of 22")
for hid, sid, hq, sq in exact_matches_22:
    print(f"  [HR{hid} -> {sid}] Exact: \"{hq}\"")

print(f"Near Matches (Jaccard >= 0.70): {len(near_matches_22)}")
for hid, sid, hq, sq, jacc in near_matches_22:
    print(f"  [HR{hid} -> {sid}] Jaccard={jacc}: \"{hq}\" <-> \"{sq}\"")

