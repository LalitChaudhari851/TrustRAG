import os
import sys
import json
import time
import subprocess
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Get current git commit hash
try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit_hash = "unknown"

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden = json.load(f)

guardrails_samples = golden.get("guardrails_samples", [])
print(f"Running guardrails eval on Phase 1 live API ({len(guardrails_samples)} samples)...")

# Capture answer text alongside blocked status
API_URL = "http://localhost:8000/query"
enriched_samples = []

for i, sample in enumerate(guardrails_samples):
    q = sample["input"]
    print(f"[{i+1}/{len(guardrails_samples)}] Testing: {q[:50]}...")
    try:
        resp = requests.post(API_URL, json={"q": q, "thread_id": f"phase1_rail_{i}"}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        thought_process = data.get("thought_process") or []
        answer = data.get("answer") or ""
        blocked = any("guardrails fired" in step.lower() for step in thought_process)
    except Exception as e:
        print(f"  Error: {e}")
        blocked = False
        answer = str(e)

    expected = sample["expected_blocked"]
    if expected and blocked:
        res = "TP"
    elif expected and not blocked:
        res = "FN"
    elif not expected and not blocked:
        res = "TN"
    else:
        res = "FP"

    enriched_samples.append({
        "id": sample["id"],
        "input": q,
        "expected_blocked": expected,
        "actual_blocked": blocked,
        "result": res,
        "answer": answer[:300]
    })
    time.sleep(3)  # Respect free tier rate limits

metrics = compute_guardrails_metrics(enriched_samples)

output = {
    "commit_hash": commit_hash,
    "date": "2026-10-16",
    "model": "qwen/qwen3.8-27b",
    "temperature": 0,
    "metrics": metrics,
    "results": enriched_samples
}

out_path = "evals/results/after_phase1_guardrails.json"
os.makedirs("evals/results", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\n=== PHASE 1 GUARDRAILS EVALUATION RESULTS ===")
print(f"Accuracy:  {metrics['accuracy']*100:.1f}% ({metrics['correct']}/{metrics['total']})")
print(f"Precision: {metrics['precision']}")
print(f"Recall:    {metrics['recall']}")
print(f"Counts:    TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
print(f"Saved to:  {out_path}")
