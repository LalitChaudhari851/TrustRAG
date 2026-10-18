import os
import sys
import json
import time
import requests
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENDPOINT_URL = "http://127.0.0.1:8000/query"
OUTPUT_JSON = "evals/results/routing_evaluation.json"

def main():
    print("=" * 80)
    print("END-TO-END ROUTING & RETRIEVAL EVALUATION THROUGH REAL /query ENDPOINT")
    print("=" * 80)

    # 1. Load Golden 15 Questions
    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_questions = [s["question"] for s in golden_raw.get("rag_samples", golden_raw)]

    # 2. Load Hard 22 Questions
    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_raw = json.load(f)
    hard_questions = [s["question"] for s in hard_raw]

    # 3. 5 Chit-chat Messages
    chitchat_messages = ["hi", "thanks", "who are you", "what can you do", "ok"]

    test_cases = []
    for idx, q in enumerate(golden_questions, 1):
        test_cases.append({"id": f"G{idx:02d}", "category": "technical", "question": q})
    for idx, q in enumerate(hard_questions, 1):
        test_cases.append({"id": f"H{idx:02d}", "category": "technical", "question": q})
    for idx, q in enumerate(chitchat_messages, 1):
        test_cases.append({"id": f"C{idx:02d}", "category": "chitchat", "question": q})

    print(f"Total Test Cases: {len(test_cases)} (15 Golden + 22 Hard + 5 Chit-Chat)")
    print("-" * 120)

    results = []
    tech_skipped = 0
    tech_retrieved = 0
    chitchat_skipped = 0
    chitchat_retrieved = 0

    header = f"{'ID':<5} | {'Category':<9} | {'Thread ID':<18} | {'Parsed Intent':<22} | {'Retrieval Ran?':<14} | {'Raw LLM Output / Query':<35} | {'Answer Snippet (100 chars)'}"
    print(header)
    print("-" * 140)

    for case in test_cases:
        cid = case["id"]
        category = case["category"]
        q = case["question"]
        thread_id = f"thread_eval_{cid}_{int(time.time()*1000)%1000000}"

        payload = {"q": q, "thread_id": thread_id}
        resp = requests.post(ENDPOINT_URL, json=payload, timeout=60)
        data = resp.json()

        thought_process = data.get("thought_process", [])
        sources = data.get("sources", [])
        answer = data.get("answer", "") or ""
        answer_snippet = answer.strip().replace("\n", " ")[:100]

        # Extract parsed intent
        parsed_intent = "Unknown"
        raw_output = "n/a"
        for item in thought_process:
            if item.startswith("Intent:"):
                parsed_intent = item.replace("Intent:", "").strip()
            if item.startswith("Raw:"):
                raw_output = item.replace("Raw:", "").strip()
            elif item.startswith("Search Term:") and raw_output == "n/a":
                raw_output = item.replace("Search Term:", "").strip()

        # Determine if retrieval ran
        retrieval_ran = (len(sources) > 0) or any("Context Retrieved" in str(x) or "Search Term:" in str(x) for x in thought_process)
        if any("Retrieval: Skipped" in str(x) for x in thought_process):
            retrieval_ran = False

        if category == "technical":
            if retrieval_ran:
                tech_retrieved += 1
            else:
                tech_skipped += 1
        else:
            if not retrieval_ran:
                chitchat_skipped += 1
            else:
                chitchat_retrieved += 1

        print(f"{cid:<5} | {category:<9} | {thread_id:<18} | {parsed_intent:<22} | {str(retrieval_ran):<14} | {raw_output[:33]:<35} | {answer_snippet}")

        results.append({
            "id": cid,
            "category": category,
            "question": q,
            "thread_id": thread_id,
            "parsed_intent": parsed_intent,
            "retrieval_ran": retrieval_ran,
            "raw_output": raw_output,
            "answer_snippet": answer_snippet,
            "status": data.get("status")
        })

        time.sleep(0.5)

    print("-" * 140)
    print("\nEVALUATION SUMMARY:")
    print(f"Technical Questions Total:      {tech_retrieved + tech_skipped}")
    print(f"Technical Questions Retrieved:  {tech_retrieved} / {tech_retrieved + tech_skipped} ({tech_retrieved/(tech_retrieved+tech_skipped)*100:.1f}%)")
    print(f"Technical Questions Skipped:    {tech_skipped} / {tech_retrieved + tech_skipped}")
    print(f"Chit-Chat Total:                {chitchat_skipped + chitchat_retrieved}")
    print(f"Chit-Chat Skipped (Correct):    {chitchat_skipped} / {chitchat_skipped + chitchat_retrieved} ({chitchat_skipped/(chitchat_skipped+chitchat_retrieved)*100:.1f}%)")
    print(f"Chit-Chat Retrieved (False):    {chitchat_retrieved}")
    total_correct = tech_retrieved + chitchat_skipped
    total_cases = len(test_cases)
    routing_correctness = total_correct / total_cases
    print(f"Overall Routing Correctness:    {total_correct} / {total_cases} ({routing_correctness*100:.2f}%)")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total_cases,
        "technical_total": tech_retrieved + tech_skipped,
        "technical_retrieved": tech_retrieved,
        "technical_skipped": tech_skipped,
        "chitchat_total": chitchat_skipped + chitchat_retrieved,
        "chitchat_skipped": chitchat_skipped,
        "chitchat_retrieved": chitchat_retrieved,
        "routing_correctness": routing_correctness,
        "cases": results
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved raw results to '{OUTPUT_JSON}'")

if __name__ == "__main__":
    main()
