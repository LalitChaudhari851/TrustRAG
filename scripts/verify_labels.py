"""
Verification script for the 22 hard retrieval questions against enterprise_rag_v2 payloads.
Checks:
  (a) expected_source exists in v2 collection
  (b) expected_section exists as a section value in v2 payloads for that source
  (c) key_term appears in that section's payload text
  (d) expected_answer is actually present in that text
Prints a detailed pass/fail table.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from qdrant_client import QdrantClient
from app.config import settings

def verify_labels():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    from qdrant_client.http import models
    # Scroll all true points from enterprise_rag_v2
    all_points = []
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name="enterprise_rag_v2",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="true"))]
            ),
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(pts)
        if next_offset is None:
            break
        offset = next_offset
    points = all_points
    
    # Index points by (source, section)
    chunks_by_src_sec = {}
    sources_in_collection = set()
    
    for p in points:
        payload = p.payload
        src = payload.get("source", "")
        sec = payload.get("section", "")
        text = payload.get("text", "")
        sources_in_collection.add(src)
        
        key = (src, sec)
        if key not in chunks_by_src_sec:
            chunks_by_src_sec[key] = []
        chunks_by_src_sec[key].append(text)
        
    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_dataset = json.load(f)
        
    print(f"\n{'='*100}")
    print(f"{'#':<3} | {'Source Exists':<14} | {'Section Exists':<15} | {'Term in Text':<13} | {'Ans in Text':<12} | {'Status':<6} | Question")
    print(f"{'='*100}")
    
    total = len(hard_dataset)
    passed = 0
    results = []
    
    for item in hard_dataset:
        qid = item["id"]
        q = item["question"]
        src = item["expected_source"]
        sec = item["expected_section"]
        term = item["key_term"]
        ans = item["expected_answer"]
        
        # Check (a)
        src_exists = src in sources_in_collection
        
        # Check (b)
        sec_exists = (src, sec) in chunks_by_src_sec
        
        # Check (c) and (d)
        matching_texts = chunks_by_src_sec.get((src, sec), [])
        combined_text = "\n".join(matching_texts)
        
        term_found = term.lower() in combined_text.lower()
        ans_found = ans.lower() in combined_text.lower()
        
        is_pass = src_exists and sec_exists and term_found and ans_found
        if is_pass:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        results.append({
            "id": qid,
            "question": q,
            "source": src,
            "section": sec,
            "src_exists": src_exists,
            "sec_exists": sec_exists,
            "term_found": term_found,
            "ans_found": ans_found,
            "status": status
        })
        
        print(f"{qid:<3} | {str(src_exists):<14} | {str(sec_exists):<15} | {str(term_found):<13} | {str(ans_found):<12} | {status:<6} | {q[:45]}...")
        if not is_pass:
            print(f"    -> [FAIL REASON] src_exists={src_exists}, sec_exists={sec_exists}, term_found={term_found}, ans_found={ans_found}")
            print(f"       Expected Source: '{src}', Section: '{sec}'")
            print(f"       Term: '{term}', Answer: '{ans}'")
            
    print(f"{'='*100}")
    print(f"Summary: {passed}/{total} verified ({passed/total*100:.1f}%)\n")
    
    with open("evals/results/hard_labels_verification.json", "w", encoding="utf-8") as f:
        json.dump({"total": total, "passed": passed, "results": results}, f, indent=2)

if __name__ == "__main__":
    verify_labels()
