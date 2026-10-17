import sys
import os
import json
import time
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails.single_call import evaluate_single_call_guard

import argparse

parser = argparse.ArgumentParser(description="Evaluate single-call input guardrails on dataset.")
parser.add_argument("--dataset", type=str, default="evals/data/guardrails_eval_22.json", help="Path to input dataset JSON")
args = parser.parse_args()

INPUT_DATASET = args.dataset
is_custom_dataset = INPUT_DATASET != "evals/data/guardrails_eval_22.json"
base_name = os.path.splitext(os.path.basename(INPUT_DATASET))[0]

JSON_OUT = f"evals/results/{base_name}_accuracy.json" if is_custom_dataset else "evals/results/single_call_accuracy.json"
TXT_OUT = f"evals/results/{base_name}_accuracy.txt" if is_custom_dataset else "evals/results/single_call_accuracy.txt"

with open(INPUT_DATASET, "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"Evaluating {len(cases)} cases from '{INPUT_DATASET}' on single_call guard...")

results = []
tp = 0
tn = 0
fp = 0
fn = 0
infra_count = 0

out_lines = []
out_lines.append("================================================================================")
out_lines.append("TASK B: SINGLE-CALL GUARD ACCURACY EVALUATION (22-CASE DEV SET)")
out_lines.append("================================================================================")
out_lines.append(f"Dataset File: {INPUT_DATASET}")
out_lines.append(f"Total Cases:  {len(cases)}")
out_lines.append(f"Prompt Version: Run 1 (Initial System Prompt - no post-result tuning)")
out_lines.append("")

out_lines.append(f"{'ID':<5} | {'Type':<12} | {'Exp Blocked':<11} | {'Guard Status':<17} | {'Decision':<8} | {'Reason':<14} | {'Match?':<6}")
out_lines.append("-" * 88)

for c in cases:
    cid = c["id"]
    c_type = c["type"]
    c_input = c["input"]
    exp_blocked = c["expected_blocked"]

    res = evaluate_single_call_guard(c_input)
    status = res["status"]       # BLOCKED | PASSED | INFRA_UNAVAILABLE
    decision = res["decision"]   # block | allow | infra_unavailable
    reason = res["reason"]       # off_topic | jailbreak | injection | unsafe_request | none | infra_error
    prompt_toks = res["prompt_tokens"]
    comp_toks = res["completion_tokens"]

    is_correct = False
    if status == "INFRA_UNAVAILABLE":
        infra_count += 1
        match_str = "INFRA"
    else:
        actual_blocked = (status == "BLOCKED")
        if exp_blocked and actual_blocked:
            tp += 1
            is_correct = True
            match_str = "TP"
        elif not exp_blocked and not actual_blocked:
            tn += 1
            is_correct = True
            match_str = "TN"
        elif not exp_blocked and actual_blocked:
            fp += 1
            match_str = "FP"
        elif exp_blocked and not actual_blocked:
            fn += 1
            match_str = "FN"

    out_lines.append(f"{cid:<5} | {c_type:<12} | {str(exp_blocked):<11} | {status:<17} | {decision:<8} | {reason:<14} | {match_str:<6}")

    results.append({
        "id": cid,
        "type": c_type,
        "description": c.get("description", ""),
        "input": c_input,
        "expected_blocked": exp_blocked,
        "guard_status": status,
        "decision": decision,
        "reason": reason,
        "classification": match_str,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks
    })

    # Pace calls slightly to prevent burst 429
    time.sleep(0.4)

non_infra_total = tp + tn + fp + fn
accuracy = (tp + tn) / non_infra_total if non_infra_total > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

out_lines.append("-" * 88)
out_lines.append("\nCONFUSION MATRIX & METRICS (EXCLUDING INFRA UNAVAILABLE CASES):")
out_lines.append(f"  Total Cases Evaluated:       {len(cases)}")
out_lines.append(f"  Infra Unavailable Cases:     {infra_count}")
out_lines.append(f"  Non-Infra Evaluated Cases:   {non_infra_total}")
out_lines.append(f"  True Positives (TP):         {tp} (Blocked correctly)")
out_lines.append(f"  True Negatives (TN):         {tn} (Allowed correctly)")
out_lines.append(f"  False Positives (FP):        {fp} (Legitimate blocked)")
out_lines.append(f"  False Negatives (FN):        {fn} (Attack allowed)")
out_lines.append("")
out_lines.append(f"  Accuracy (non-infra):        {accuracy:.4f} ({tp+tn}/{non_infra_total})")
out_lines.append(f"  Precision:                   {precision:.4f} ({tp}/{tp+fp})")
out_lines.append(f"  Recall:                      {recall:.4f} ({tp}/{tp+fn})")
out_lines.append(f"  F1 Score:                    {f1:.4f}")

full_text = "\n".join(out_lines) + "\n"

os.makedirs(os.path.dirname(TXT_OUT), exist_ok=True)
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(full_text)

json_payload = {
    "script": "scripts/run_single_call_accuracy.py",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_cases": len(cases),
    "infra_count": infra_count,
    "non_infra_total": non_infra_total,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cases": results
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)

print(full_text)
