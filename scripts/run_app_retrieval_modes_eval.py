import os
import sys
import json
import time
import requests
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENDPOINT_URL = "http://127.0.0.1:8000/query"
OUTPUT_JSON = "evals/results/app_retrieval_modes_eval.json"

NO_INFO_PHRASES = [
    "no information",
    "not have enough context",
    "cannot find",
    "does not mention",
    "not mentioned",
    "does not provide",
    "not provided in the",
    "no context"
]

def is_no_info(answer: str) -> bool:
    ans_lower = answer.lower()
    return any(p in ans_lower for p in NO_INFO_PHRASES)

def is_rate_limit(answer: str) -> bool:
    ans_lower = answer.lower()
    return (
        "rate limit" in ans_lower
        or "429" in ans_lower
        or "high demand" in ans_lower
        or "safety check temporarily unavailable" in ans_lower
    )

def load_all_questions():
    test_cases = []
    
    # 1. Golden 15
    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    samples = golden_raw.get("rag_samples", golden_raw)
    for idx, s in enumerate(samples, 1):
        kp = s.get("key_phrase")
        if not kp:
            ref = s.get("reference", "")
            kp = ref[:30]
        test_cases.append({
            "id": f"G{idx:02d}",
            "question": s["question"],
            "key_phrase": kp,
            "source_dataset": "golden_dataset.json"
        })

    # 2. Hard 22
    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_raw = json.load(f)
    for idx, s in enumerate(hard_raw, 1):
        test_cases.append({
            "id": f"H{idx:02d}",
            "question": s["question"],
            "key_phrase": s.get("key_term", ""),
            "source_dataset": "hard_retrieval_22.json"
        })

    # 3. Held-out questions if present
    heldout_path = "evals/data/e2e_heldout.json"
    heldout_found = False
    if os.path.exists(heldout_path):
        heldout_found = True
        with open(heldout_path, "r", encoding="utf-8") as f:
            h_raw = json.load(f)
        for idx, s in enumerate(h_raw, 1):
            test_cases.append({
                "id": f"X{idx:02d}",
                "question": s["question"],
                "key_phrase": s.get("key_term", s.get("key_phrase", "")),
                "source_dataset": "e2e_heldout.json"
            })
    else:
        print(f"Notice: Heldout file '{heldout_path}' is missing, skipping heldout evaluation.", flush=True)

    return test_cases, heldout_found

def run_eval_for_mode(mode: str, test_cases: list):
    print(f"\n================================================================================", flush=True)
    print(f"RUNNING EVALUATION FOR RETRIEVAL_MODE = '{mode.upper()}' ({len(test_cases)} questions)", flush=True)
    print(f"================================================================================", flush=True)

    results = []
    
    for case in test_cases:
        cid = case["id"]
        q = case["question"]
        kp = case["key_phrase"]
        tid = f"thread_eval_{mode}_{cid}_{int(time.time()*1000)%1000000}"

        payload = {
            "q": q,
            "thread_id": tid,
            "retrieval_mode": mode
        }

        # Query with automatic retry on 429
        max_attempts = 5
        resp_data = {}
        for attempt in range(1, max_attempts + 1):
            try:
                r = requests.post(ENDPOINT_URL, json=payload, timeout=60)
                resp_data = r.json()
                ans = resp_data.get("answer", "")
                if not is_rate_limit(ans):
                    break
                wait_sec = attempt * 4.0
                print(f"  [Attempt {attempt}] Rate limit / guardrails 429 detected for {cid}. Waiting {wait_sec:.1f}s before retry...", flush=True)
                time.sleep(wait_sec)
            except Exception as e:
                wait_sec = attempt * 3.0
                print(f"  [Attempt {attempt}] Request error for {cid}: {e}. Waiting {wait_sec:.1f}s...", flush=True)
                time.sleep(wait_sec)

        answer = resp_data.get("answer", "")
        sources = resp_data.get("sources", [])
        thought = resp_data.get("thought_process", [])

        # Extract planner query from thought process
        planner_query = q  # Default raw
        for step in thought:
            if "Search Term:" in step:
                planner_query = step.split("Search Term:", 1)[1].strip()
            elif "Raw:" in step:
                raw_part = step.split("Raw:", 1)[1].strip()
                if "query" in raw_part:
                    try:
                        parsed_raw = json.loads(raw_part)
                        if parsed_raw.get("query"):
                            planner_query = parsed_raw.get("query")
                    except Exception:
                        pass

        # Check key phrase matches (case-insensitive substring)
        kp_lower = kp.lower()
        context_str = " ".join([s if isinstance(s, str) else s.get("content", "") for s in sources]).lower()
        kp_in_ctx = bool(kp_lower in context_str) if kp else False
        kp_in_ans = bool(kp_lower in answer.lower()) if kp else False
        no_info = is_no_info(answer)
        rate_limited = is_rate_limit(answer)
        retrieval_ran = len(sources) > 0 or any("Context" in t for t in thought)
        ans_produced = bool(answer and not rate_limited)

        res_item = {
            "id": cid,
            "question": q,
            "key_phrase": kp,
            "thread_id": tid,
            "planner_query": planner_query,
            "answer_snippet": answer[:100].replace("\n", " "),
            "retrieval_invoked": retrieval_ran,
            "answer_produced": ans_produced,
            "is_rate_limit_error": rate_limited,
            "key_phrase_in_context": kp_in_ctx,
            "key_phrase_in_answer": kp_in_ans,
            "is_no_info_answer": no_info,
            "sources_count": len(sources),
            "full_answer": answer
        }
        results.append(res_item)

        clean_ans_preview = answer[:60].replace("\n", " ").encode("ascii", "ignore").decode("ascii")
        print(f"  {cid} | InCtx: {str(kp_in_ctx):<5} | InAns: {str(kp_in_ans):<5} | NoInfo: {str(no_info):<5} | RateLim: {str(rate_limited):<5} | Ans: {clean_ans_preview}...", flush=True)
        time.sleep(2.0)  # Pace queries to respect TPM limit

    return results

def main():
    test_cases, heldout_found = load_all_questions()
    print(f"Loaded {len(test_cases)} evaluation cases.")

    modes = ["tinybert", "rrf", "fusion"]
    all_mode_results = {}

    for m in modes:
        all_mode_results[m] = run_eval_for_mode(m, test_cases)

        # Save partial progress after each mode
        summary = {}
        for done_m in all_mode_results:
            res = all_mode_results[done_m]
            tot = len(res)
            ret_invoked = sum(1 for r in res if r["retrieval_invoked"])
            ans_prod = sum(1 for r in res if r["answer_produced"])
            rate_errs = sum(1 for r in res if r["is_rate_limit_error"])
            kp_ctx = sum(1 for r in res if r["key_phrase_in_context"])
            kp_ans = sum(1 for r in res if r["key_phrase_in_answer"])
            no_info_cnt = sum(1 for r in res if r["is_no_info_answer"])

            h03_item = next((r for r in res if r["id"] == "H03"), None)
            h04_item = next((r for r in res if r["id"] == "H04"), None)

            summary[done_m] = {
                "total_questions": tot,
                "retrieval_invoked": ret_invoked,
                "answer_produced": ans_prod,
                "rate_limit_errors": rate_errs,
                "key_phrase_in_context": kp_ctx,
                "key_phrase_in_context_pct": kp_ctx / tot * 100.0 if tot > 0 else 0,
                "key_phrase_in_answer": kp_ans,
                "key_phrase_in_answer_pct": kp_ans / tot * 100.0 if tot > 0 else 0,
                "no_info_answers": no_info_cnt,
                "H03": {
                    "key_phrase_in_context": h03_item["key_phrase_in_context"] if h03_item else None,
                    "key_phrase_in_answer": h03_item["key_phrase_in_answer"] if h03_item else None,
                    "is_no_info_answer": h03_item["is_no_info_answer"] if h03_item else None,
                    "answer_snippet": h03_item["answer_snippet"] if h03_item else None
                },
                "H04": {
                    "key_phrase_in_context": h04_item["key_phrase_in_context"] if h04_item else None,
                    "key_phrase_in_answer": h04_item["key_phrase_in_answer"] if h04_item else None,
                    "is_no_info_answer": h04_item["is_no_info_answer"] if h04_item else None,
                    "answer_snippet": h04_item["answer_snippet"] if h04_item else None
                }
            }

        output_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "heldout_file_present": heldout_found,
            "modes_summary": summary,
            "details_per_mode": all_mode_results
        }
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Updated '{OUTPUT_JSON}' with completed mode: {m}", flush=True)

    print(f"\nCompleted all modes. Saved evaluation results to '{OUTPUT_JSON}'", flush=True)

if __name__ == "__main__":
    main()
