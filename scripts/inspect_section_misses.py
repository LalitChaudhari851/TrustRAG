import sys, os, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('evals/results/retrieval_ab_full.json', encoding='utf-8') as f:
    d = json.load(f)

with open('evals/golden_dataset.json', encoding='utf-8') as f:
    golden_raw = json.load(f)
golden_map = {q['id']: q for q in golden_raw.get('rag_samples', golden_raw)}

with open('evals/data/hard_retrieval_22.json', encoding='utf-8') as f:
    hard_raw = json.load(f)
hard_map = {q['id']: q for q in hard_raw}

print("==================================================")
print("SECTION MISSES IN RETRIEVAL A/B (HYBRID ON FULL V2)")
print("==================================================")

miss_records = []

# 1. Golden 15
print("\n--- GOLDEN 15 SECTION MISSES ---")
for q in d['golden_full_v2']['v2_full_hybrid']['queries']:
    if not q['section_hit_5']:
        qid = q['id']
        g_meta = golden_map.get(qid, {})
        expected_sec = q['expected_section']
        retrieved_secs = q['top_5_sections']
        retrieved_srcs = q['top_5_sources']
        
        # Check text of top-5 chunks against expected key answers
        expected_ref = g_meta.get('reference', '')
        # Check relevant_contexts
        rel_contexts = g_meta.get('relevant_contexts', [])
        
        record = {
            "dataset": "Golden 15",
            "id": qid,
            "question": q['question'],
            "expected_source": q['expected_source'],
            "expected_section": expected_sec,
            "retrieved_top1_source": retrieved_srcs[0] if retrieved_srcs else "",
            "retrieved_top1_section": retrieved_secs[0] if retrieved_secs else "",
            "retrieved_sections_top5": retrieved_secs,
            "expected_reference": expected_ref
        }
        miss_records.append(record)
        print(f"\n[Golden Q{qid}] {q['question']}")
        print(f"  Expected Source:  {q['expected_source']}")
        print(f"  Expected Section: {expected_sec}")
        print(f"  Retrieved Top-1:  {retrieved_srcs[0]} -> {retrieved_secs[0]}")
        print(f"  Retrieved Top-5 Sections: {retrieved_secs}")

# 2. Hard 22
print("\n--- HARD 22 SECTION MISSES ---")
for q in d['hard_full_v2']['v2_full_hybrid']['queries']:
    if not q['section_hit_5']:
        qid = q['id']
        h_meta = hard_map.get(qid, {})
        expected_sec = q['expected_section']
        retrieved_secs = q['top_5_sections']
        retrieved_srcs = q['top_5_sources']
        key_term = h_meta.get('key_term', '')
        expected_ans = h_meta.get('expected_answer', '')
        
        record = {
            "dataset": "Hard 22",
            "id": qid,
            "question": q['question'],
            "expected_source": q['expected_source'],
            "expected_section": expected_sec,
            "retrieved_top1_source": retrieved_srcs[0] if retrieved_srcs else "",
            "retrieved_top1_section": retrieved_secs[0] if retrieved_secs else "",
            "retrieved_sections_top5": retrieved_secs,
            "key_term": key_term,
            "expected_answer": expected_ans
        }
        miss_records.append(record)
        print(f"\n[Hard Q{qid}] {q['question']}")
        print(f"  Expected Source:  {q['expected_source']}")
        print(f"  Expected Section: {expected_sec}")
        print(f"  Retrieved Top-1:  {retrieved_srcs[0]} -> {retrieved_secs[0]}")
        print(f"  Retrieved Top-5 Sections: {retrieved_secs}")
        print(f"  Verified Key Term: {key_term}")

with open("evals/results/section_misses_analysis.json", "w", encoding="utf-8") as f:
    json.dump(miss_records, f, indent=2)

print("\nSaved section misses analysis to evals/results/section_misses_analysis.json")
