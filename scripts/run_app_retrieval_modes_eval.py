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

from scripts.run_retrieval_ab import GOLDEN_KEY_TERMS

ENDPOINT_URL = "http://127.0.0.1:8000/query"
CONFIG_URL = "http://127.0.0.1:8000/config"
OUTPUT_JSON = "evals/results/app_retrieval_modes_eval.json"

NO_INFO_PHRASES = [
    "no information",
    "not have enough context",
    "cannot find",
    "does not mention",
    "not mentioned",
    "does not provide",
    "not provided in the",
    "no context",
    "not found in the documentation"
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

def fetch_server_config() -> Dict[str, Any]:
    try:
        r = requests.get(CONFIG_URL, timeout=10)
        return r.json()
    except Exception as e:
        print(f"Warning: Could not fetch server config from {CONFIG_URL}: {e}", flush=True)
        return {}

def load_all_questions():
    test_cases = []
    
    # 1. Golden 15 (using ground truth key terms from GOLDEN_KEY_TERMS)
    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    samples = golden_raw.get("rag_samples", golden_raw)
    for idx, s in enumerate(samples, 1):
        kp = GOLDEN_KEY_TERMS.get(idx) or s.get("key_phrase") or s.get("reference", "")[:30]
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

    # 3. Held-out questions if present (without reading or modifying missing files)
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

def execute_single_query(cid: str, q: str, mode: str, max_attempts: int = 5) -> Dict[str, Any]:
    tid = f"thread_eval_{mode}_{cid}_{int(time.time()*1000)%1000000}"
    payload = {
        "q": q,
        "thread_id": tid,
        "retrieval_mode": mode
    }

    resp_data = {}
    is_rl = False
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(ENDPOINT_URL, json=payload, timeout=60)
            resp_data = r.json()
            ans = resp_data.get("answer", "")
            if not is_rate_limit(ans):
                is_rl = False
                break
            is_rl = True
            wait_sec = attempt * 4.0
            print(f"  [Attempt {attempt}] 429 rate limit for {cid}. Waiting {wait_sec:.1f}s before retry...", flush=True)
            time.sleep(wait_sec)
        except Exception as e:
            is_rl = False
            wait_sec = attempt * 3.0
            print(f"  [Attempt {attempt}] Connection error for {cid}: {e}. Waiting {wait_sec:.1f}s...", flush=True)
            time.sleep(wait_sec)

    return resp_data, is_rl, tid

def run_eval_for_mode(mode: str, test_cases: list):
    print(f"\n================================================================================", flush=True)
    print(f"RUNNING EVALUATION FOR RETRIEVAL_MODE = '{mode.upper()}' ({len(test_cases)} questions)", flush=True)
    print(f"================================================================================", flush=True)

    results = []
    rate_limit_errors = []
    
    for case in test_cases:
        cid = case["id"]
        q = case["question"]
        kp = case["key_phrase"]

        resp_data, is_rl, tid = execute_single_query(cid, q, mode)
        if is_rl:
            rate_limit_errors.append(cid)

        answer = resp_data.get("answer", "")
        sources = resp_data.get("sources", [])
        thought = resp_data.get("thought_process", [])
        mode_used = resp_data.get("retrieval_mode", mode)

        # Extract planner query from thought process
        planner_query = q
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

        # Check key phrase matches
        kp_lower = kp.lower() if kp else ""
        context_str = " ".join([s if isinstance(s, str) else s.get("content", "") for s in sources]).lower()
        kp_in_ctx = bool(kp_lower in context_str) if kp_lower else False
        kp_in_ans = bool(kp_lower in answer.lower()) if kp_lower else False
        
        # Groundedness flag: answer mentions key phrase without supporting context
        answer_without_supporting_context = bool(kp_in_ans and not kp_in_ctx)

        # Retrieval invoked check
        retrieval_invoked = bool(sources and len(sources) > 0)
        if any("Retrieval: Skipped" in st for st in thought) or any("Guardrails Fired" in st for st in thought):
            retrieval_invoked = False

        answer_produced = bool(answer and len(answer.strip()) > 0 and not is_rl)
        no_info = is_no_info(answer)

        model_used = resp_data.get("model_used", "openai/gpt-oss-120b")

        res_record = {
            "id": cid,
            "question": q,
            "key_phrase": kp,
            "thread_id": tid,
            "planner_query": planner_query,
            "retrieval_mode_requested": mode,
            "retrieval_mode_used": mode_used,
            "model_used": model_used,
            "answer_snippet": answer[:100].replace("\n", " ") if answer else "",
            "retrieval_invoked": retrieval_invoked,
            "answer_produced": answer_produced,
            "is_rate_limit_error": is_rl,
            "key_phrase_in_context": kp_in_ctx,
            "key_phrase_in_answer": kp_in_ans,
            "answer_without_supporting_context": answer_without_supporting_context,
            "is_no_info_answer": no_info,
            "sources_count": len(sources),
            "full_answer": answer
        }
        results.append(res_record)

        status_str = f"ctx={'YES' if kp_in_ctx else 'NO '} | ans={'YES' if kp_in_ans else 'NO '} | ungrounded={'YES' if answer_without_supporting_context else 'NO '}"
        print(f"[{cid}] mode={mode} | {status_str} | sources={len(sources)} | q: {q[:45]}...", flush=True)
        time.sleep(2.0)  # Paced pacing to avoid 429 rate limit errors

    # Re-run any rate-limit errors if any occurred
    if rate_limit_errors:
        print(f"\nRetrying {len(rate_limit_errors)} rate-limit errors after a 10s cool-down...", flush=True)
        time.sleep(10.0)
        for i, res in enumerate(results):
            if res["id"] in rate_limit_errors:
                cid = res["id"]
                resp_data, is_rl, tid = execute_single_query(cid, res["question"], mode, max_attempts=5)
                if not is_rl:
                    answer = resp_data.get("answer", "")
                    sources = resp_data.get("sources", [])
                    context_str = " ".join([s if isinstance(s, str) else s.get("content", "") for s in sources]).lower()
                    kp_lower = res["key_phrase"].lower() if res["key_phrase"] else ""
                    res["key_phrase_in_context"] = bool(kp_lower in context_str) if kp_lower else False
                    res["key_phrase_in_answer"] = bool(kp_lower in answer.lower()) if kp_lower else False
                    res["answer_without_supporting_context"] = bool(res["key_phrase_in_answer"] and not res["key_phrase_in_context"])
                    res["is_rate_limit_error"] = False
                    res["answer_produced"] = True
                    res["sources_count"] = len(sources)
                    res["full_answer"] = answer
                    res["answer_snippet"] = answer[:100].replace("\n", " ")
                    print(f"  Retry succeeded for {cid}!", flush=True)

    return results, rate_limit_errors

def main():
    print("=== MULTI-MODE APP RETRIEVAL EVALUATION (TINYBERT / RRF / FUSION) ===", flush=True)
    
    server_cfg = fetch_server_config()
    print(f"Live Server Runtime Config: {json.dumps(server_cfg, indent=2)}", flush=True)

    test_cases, heldout_found = load_all_questions()
    print(f"Total test cases to evaluate per mode: {len(test_cases)}", flush=True)

    modes = ["tinybert", "rrf", "fusion"]
    modes_results = {}
    modes_summary = {}

    for mode in modes:
        mode_records, mode_rl_errors = run_eval_for_mode(mode, test_cases)
        modes_results[mode] = mode_records

        tot = len(mode_records)
        invoked = sum(1 for r in mode_records if r["retrieval_invoked"])
        produced = sum(1 for r in mode_records if r["answer_produced"])
        rl_count = sum(1 for r in mode_records if r["is_rate_limit_error"])
        kp_ctx = sum(1 for r in mode_records if r["key_phrase_in_context"])
        kp_ans = sum(1 for r in mode_records if r["key_phrase_in_answer"])
        ungrounded = sum(1 for r in mode_records if r["answer_without_supporting_context"])
        no_info = sum(1 for r in mode_records if r["is_no_info_answer"])

        h03_record = next((r for r in mode_records if r["id"] == "H03"), {})
        h04_record = next((r for r in mode_records if r["id"] == "H04"), {})

        modes_summary[mode] = {
            "total_questions": tot,
            "retrieval_invoked": invoked,
            "answer_produced": produced,
            "rate_limit_errors": rl_count,
            "key_phrase_in_context": kp_ctx,
            "key_phrase_in_context_pct": round(kp_ctx / tot * 100, 2) if tot else 0,
            "key_phrase_in_answer": kp_ans,
            "key_phrase_in_answer_pct": round(kp_ans / tot * 100, 2) if tot else 0,
            "answer_without_supporting_context": ungrounded,
            "no_info_answers": no_info,
            "H03": {
                "key_phrase_in_context": h03_record.get("key_phrase_in_context"),
                "key_phrase_in_answer": h03_record.get("key_phrase_in_answer"),
                "is_no_info_answer": h03_record.get("is_no_info_answer"),
                "answer_snippet": h03_record.get("answer_snippet")
            },
            "H04": {
                "key_phrase_in_context": h04_record.get("key_phrase_in_context"),
                "key_phrase_in_answer": h04_record.get("key_phrase_in_answer"),
                "is_no_info_answer": h04_record.get("is_no_info_answer"),
                "answer_snippet": h04_record.get("answer_snippet")
            }
        }

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_runtime_config": server_cfg,
        "heldout_file_present": heldout_found,
        "modes_summary": modes_summary,
        "details_per_mode": modes_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n================================================================================", flush=True)
    print("FINAL MULTI-MODE APP RETRIEVAL SUMMARY TABLE", flush=True)
    print("================================================================================", flush=True)
    print(f"{'Mode':<10} | {'Total':<6} | {'Invoked':<8} | {'KP in Ctx':<10} | {'KP in Ans':<10} | {'Ungrounded':<10} | {'429 Errs':<8}")
    print("-" * 75)
    for mode, s in modes_summary.items():
        print(f"{mode:<10} | {s['total_questions']:<6} | {s['retrieval_invoked']:<8} | {s['key_phrase_in_context']:<10} | {s['key_phrase_in_answer']:<10} | {s['answer_without_supporting_context']:<10} | {s['rate_limit_errors']:<8}")

    print(f"\nSaved full results to {OUTPUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
