import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails.single_call import evaluate_single_call_guard
from app.config import settings

DEV_SET_PATH = "evals/data/guardrails_eval_22.json"
OUTPUT_PATH = "evals/results/guardrails_dev_22_results.json"

def main():
    print("=== EVALUATING 22-CASE GUARDRAILS DEV SET ===")
    print("Label: dev set, prompt edited after seeing G03 failure; not an unbiased estimate")
    
    with open(DEV_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    records = []
    correct_count = 0
    
    for case in cases:
        cid = case["id"]
        text = case["input"]
        exp_blocked = case["expected_blocked"]
        q_type = case.get("type", "")
        
        # Call guard
        res = evaluate_single_call_guard(text)
        is_blocked = (res["decision"] == "block")
        is_correct = (is_blocked == exp_blocked)
        if is_correct:
            correct_count += 1
            
        records.append({
            "id": cid,
            "input": text,
            "type": q_type,
            "expected_blocked": exp_blocked,
            "actual_decision": res["decision"],
            "actual_reason": res["reason"],
            "is_blocked": is_blocked,
            "is_correct": is_correct,
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "model_used": res.get("model_used", "")
        })
        print(f"[{cid}] exp={exp_blocked} actual={res['decision']} ({res['reason']}) -> {'PASS' if is_correct else 'FAIL'}")
        time.sleep(1.0)  # Paced to avoid rate limits
        
    accuracy = correct_count / len(cases)
    
    result_payload = {
        "label": "dev set, prompt edited after seeing G03 failure; not an unbiased estimate",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(cases),
        "correct": correct_count,
        "accuracy": accuracy,
        "model_used": records[0]["model_used"] if records else "unknown",
        "guard_system_prompt_scope": "Derived directly from CORPUS (Kubernetes, Redis, Databricks jobs, OS/kernel, C++, compilers, networking, ML papers, databases, security-defense questions)",
        "cases": records
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
        
    print(f"\nCompleted 22-case dev set: {correct_count}/{len(cases)} correct ({accuracy*100:.1f}%). Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
