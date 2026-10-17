import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)["rag_samples"]

terms_to_check = {
    3: "rpush job2",
    10: "2023-05-10",
    15: "kubectl cluster-info"
}

# Fetch all true_data chunks (or scroll through all points)
scroll_pts = []
next_offset = None
while True:
    records, next_offset = client.scroll(
        collection_name="enterprise_rag_v2",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="true"))]
        ),
        limit=250,
        with_payload=True,
        offset=next_offset
    )
    scroll_pts.extend(records)
    if next_offset is None:
        break

print(f"Scrolled {len(scroll_pts)} true chunks from enterprise_rag_v2.")

out_lines = []
out_lines.append("================================================================================")
out_lines.append("TASK 2: GOLDEN KEY PHRASES ANALYSIS FOR Q3, Q10, Q15")
out_lines.append("================================================================================")

for qid, term in terms_to_check.items():
    q_obj = next(q for q in golden_data if q["id"] == qid)
    out_lines.append(f"\n--- Golden Query Q{qid} ---")
    out_lines.append(f"Question: \"{q_obj['question']}\"")
    out_lines.append(f"Expected Source:  {q_obj.get('expected_source')}")
    out_lines.append(f"Expected Section: {q_obj.get('expected_section')}")
    out_lines.append(f"Current Key Phrase in Code: \"{term}\"")
    out_lines.append(f"Reference Answer: \"{q_obj.get('reference')}\"")

    # Grep in all true chunks
    matching_chunks = [p for p in scroll_pts if term.lower() in p.payload.get("text", "").lower()]
    out_lines.append(f"Occurrences of \"{term}\" across ALL 199 true chunks in Qdrant: {len(matching_chunks)}")

    # Also search chunks from the expected source and section
    expected_src = q_obj.get("expected_source", "")
    target_chunks = [p for p in scroll_pts if p.payload.get("source") == expected_src]
    
    out_lines.append(f"Inspecting real chunks from expected source '{expected_src}':")
    for idx, tc in enumerate(target_chunks[:5], 1):
        sec = tc.payload.get("section", "")
        txt = tc.payload.get("text", "")
        out_lines.append(f"  Chunk {idx} [{expected_src} -> {sec}]:")
        out_lines.append(f"    Preview (200 chars): {txt[:200]!r}")

    # Propose corrected key phrase
    if qid == 3:
        # Check for redis CLI queue filling in parallel_work_queue.txt
        out_lines.append("\n  [ANALYSIS & PROPOSAL FOR Q3]:")
        out_lines.append("  Question asks: 'How do you fill the Redis work queue with tasks using the CLI?'")
        # Let's inspect target_chunks for matches
        for tc in target_chunks:
            if "queue" in tc.payload.get("text", "").lower() or "redis" in tc.payload.get("text", "").lower():
                out_lines.append(f"    Potential Match [{tc.payload.get('section')}]: {tc.payload.get('text')[:300]!r}")
    elif qid == 10:
        out_lines.append("\n  [ANALYSIS & PROPOSAL FOR Q10]:")
        out_lines.append("  Question asks: 'What are the valid restart policy values for a Kubernetes Job pod?'")
        out_lines.append("  '2023-05-10' appears in 0 chunks and is a clear copy-paste date bug.")
        for tc in target_chunks:
            if "restart" in tc.payload.get("text", "").lower() or "onfailure" in tc.payload.get("text", "").lower():
                out_lines.append(f"    Target restart chunk [{tc.payload.get('section')}]: {tc.payload.get('text')[:300]!r}")
    elif qid == 15:
        out_lines.append("\n  [ANALYSIS & PROPOSAL FOR Q15]:")
        out_lines.append("  Question asks: 'How do you verify that the core Kubernetes components are running after starting Minikube?'")
        for tc in target_chunks:
            if "kube-system" in tc.payload.get("text", "").lower() or "minikube" in tc.payload.get("text", "").lower():
                out_lines.append(f"    Target verification chunk [{tc.payload.get('section')}]: {tc.payload.get('text')[:300]!r}")

res_text = "\n".join(out_lines) + "\n"
print(res_text)
with open("evals/results/golden_key_phrases_analysis.txt", "w", encoding="utf-8") as f:
    f.write(res_text)
