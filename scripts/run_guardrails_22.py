import os
import sys
import json
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "http://localhost:8000/query"

with open("evals/data/guardrails_eval_22.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"=== RUNNING 22-CASE GUARDRAILS EVALUATION ===")
print(f"Total samples: {len(samples)}")

results = []
for i, sample in enumerate(samples):
    q = sample["input"]
    expected = sample["expected_blocked"]
    print(f"[{i+1:02d}/{len(samples)}] Testing {sample['id']} ({sample['type']}): {q[:45]}...", end=" ", flush=True)
    try:
        resp = requests.post(API_URL, json={"q": q, "thread_id": f"guard_eval_22_{i}"}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        tp = data.get("thought_process") or []
        answer = data.get("answer") or ""
        blocked = any("guardrails fired" in step.lower() for step in tp)
    except Exception as e:
        print(f"ERR: {e}")
        blocked = False
        answer = str(e)

    if expected and blocked:
        res = "TP"
    elif expected and not blocked:
        res = "FN"
    elif not expected and not blocked:
        res = "TN"
    else:
        res = "FP"

    match = "PASS" if res in ("TP", "TN") else "FAIL"
    print(f"-> [{match}] {res} (blocked={blocked})")

    results.append({
        "id": sample["id"],
        "input": q,
        "type": sample["type"],
        "expected_blocked": expected,
        "actual_blocked": blocked,
        "result": res,
        "answer": answer[:200]
    })
    # Respect Groq 7000 ITPM limit
    time.sleep(3)

tp_cnt = sum(1 for r in results if r["result"] == "TP")
tn_cnt = sum(1 for r in results if r["result"] == "TN")
fp_cnt = sum(1 for r in results if r["result"] == "FP")
fn_cnt = sum(1 for r in results if r["result"] == "FN")

acc = (tp_cnt + tn_cnt) / len(results)
prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
rec = tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else 0.0

summary = {
    "model": "qwen/qwen3.8-27b",
    "total": len(results),
    "correct": tp_cnt + tn_cnt,
    "accuracy": round(acc, 3),
    "precision": round(prec, 3),
    "recall": round(rec, 3),
    "tp": tp_cnt, "tn": tn_cnt, "fp": fp_cnt, "fn": fn_cnt,
    "results": results
}

dest = "evals/results/guardrails_eval_22_results.json"
with open(dest, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== EVALUATION COMPLETE ===")
print(f"Accuracy:  {acc*100:.1f}% ({tp_cnt + tn_cnt}/{len(results)})")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"Confusion: TP={tp_cnt}, TN={tn_cnt}, FP={fp_cnt}, FN={fn_cnt}")
print(f"Saved to:  {dest}")
