import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.retrieval.qdrant_service import client, search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

SWEEP_FILE = "evals/results/threshold_sweep_full.json"
COLLECTION_NAME = "enterprise_rag_v2"
OUTPUT_FILE = "evals/results/verification_3.txt"

with open(SWEEP_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

per_query = data.get("per_query_details", {})
k8s_9 = per_query.get("negatives_k8s_9", [])
noisy_9 = per_query.get("negatives_noisy_9", [])

def get_top_chunk_preview(query: str) -> str:
    try:
        retrieved = search_enterprise_knowledge(query, limit=15, hybrid=True)
        if not retrieved:
            return "[NO RETRIEVED CHUNKS]"
        reranked = rerank_documents(query, retrieved, top_n=1, min_score=0.0, relative_margin=1.0)
        if not reranked:
            return "[NO RERANKED CHUNKS]"
        top = reranked[0]
        text = (top.get("content") or top.get("text") or "").replace("\n", " ").strip()
        return text[:100]
    except Exception as e:
        return f"[ERROR: {e}]"

lines = []
lines.append("================================================================================")
lines.append("VERIFICATION ITEM 3: 18 HARD NEGATIVES STRAIGHT FROM threshold_sweep_full.json")
lines.append("================================================================================")
lines.append(f"Ranker Model: {data.get('ranker_model')}")
lines.append(f"Collection:   {data.get('collection')}")
lines.append(f"Timestamp:    {data.get('timestamp')}")

lines.append("\n" + "="*80)
lines.append("GROUP 1: negatives_k8s_9 (9 In-Domain Kubernetes Hard Negatives)")
lines.append("Definition: Highly specific Kubernetes orchestration questions regarding concepts (e.g. Istio,")
lines.append("Ingress TLS, NetworkPolicy CIDR, PodDisruptionBudget, CSI, CRDs) that are plausible Kubernetes")
lines.append("queries but are completely ABSENT from the 6 ingested enterprise true_data documents.")
lines.append("="*80)

for idx, item in enumerate(k8s_9, 1):
    q = item["question"]
    preview = get_top_chunk_preview(q)
    lines.append(f"\n[Neg {idx:02d} - K8S] Question: {q}")
    lines.append(f"  Top Score:       {item['top_score']}")
    lines.append(f"  Top Source:      {item['top_source']} (source_type: {item.get('top_source_type', 'unknown')})")
    lines.append(f"  Top Section:     {item['top_section']}")
    lines.append(f"  Top-5 Scores:    {item.get('all_scores', [])}")
    lines.append(f"  100-char Chunk:  {preview}...")

lines.append("\n" + "="*80)
lines.append("GROUP 2: negatives_noisy_9 (9 Out-of-Domain / Noisy Corpus Hard Negatives)")
lines.append("Definition: Specialized computer science and systems queries (Linux kernel paging, C++ stack unwinding,")
lines.append("SerDes optical framing, Burrows-Wheeler, SAL buffer overflow, etc.) representing content from the")
lines.append("noisy_data distraction corpus to test whether cross-domain technical queries trigger false retrieval.")
lines.append("="*80)

for idx, item in enumerate(noisy_9, 10):
    q = item["question"]
    preview = get_top_chunk_preview(q)
    lines.append(f"\n[Neg {idx:02d} - NOISY] Question: {q}")
    lines.append(f"  Top Score:       {item['top_score']}")
    lines.append(f"  Top Source:      {item['top_source']} (source_type: {item.get('top_source_type', 'unknown')})")
    lines.append(f"  Top Section:     {item['top_section']}")
    lines.append(f"  Top-5 Scores:    {item.get('all_scores', [])}")
    lines.append(f"  100-char Chunk:  {preview}...")

lines.append("\n" + "="*80)
lines.append("RECONCILIATION: '3/18 REJECTED' AND MEAN CHUNKS PASSED (3.43 vs 3.8)")
lines.append("="*80)
lines.append("1. Reconciling '3/18 Rejected' with Score Floor (0.08):")
lines.append("   - In threshold_sweep_full.json, at floor=0.08 and margin=0.20:")
lines.append("     * true_rejections = 3 out of 18 (16.67%).")
lines.append("     * The 3 rejected negatives are:")
lines.append("       1) HN_K8S_7 ('PersistentVolume dynamic provisioners'): Top Score = 0.0001 < 0.08 (REJECTED)")
lines.append("       2) HN_NOISY_3 ('Burrows-Wheeler block-sorting'): Top Score = 0.0796 < 0.08 (REJECTED)")
lines.append("       3) HN_NOISY_7 ('Transactional memory HTM'): Top Score = 0.0796 < 0.08 (REJECTED)")
lines.append("     * The remaining 15 negatives received top rerank scores between 0.12 and 0.9998 because")
lines.append("       the cross-encoder (TinyBERT) experiences lexical score inflation when technical terms")
lines.append("       match generic words in the corpus (e.g. 'Kubernetes', 'data', 'buffer').")
lines.append("     * Therefore, a score floor alone cannot reject in-domain negatives without upstream guardrails.")
lines.append("")
lines.append("2. Reconciling Mean Chunks Passed (3.43 vs 3.8):")
lines.append("   - In threshold_sweep_full.json, the measured 'avg_chunks_passed_when_accepted' across the sweep grid is:")
lines.append("     * Margin 0.20: 3.43 chunks")
lines.append("     * Margin 0.30: 3.57 chunks")
lines.append("     * Margin 0.40: 3.65 chunks")
lines.append("     * Margin 0.50: 3.73 chunks")
lines.append("   - The value '3.8' in the previous narrative report was NOT generated by code. It was an unverified,")
lines.append("     rounded estimate typed directly into the report summary by the assistant.")
lines.append("   - The actual ground-truth measured value from run_threshold_sweep.py is 3.43 chunks passed (at margin 0.20).")

full_content = "\n".join(lines) + "\n"
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write(full_content)

print(full_content)
