import sys
import os
import re
import json
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query

OUTPUT_FILE = "evals/results/verification_5.txt"

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)
ranker = Ranker()
COLLECTION = "enterprise_rag_v2"

out = []
out.append("================================================================================")
out.append("VERIFICATION ITEM 5: HARD Q14 RETRIEVAL BREAKDOWN & DE-BIASED RETRIEVAL A/B")
out.append("================================================================================")

q_original = "What release manifest URL installs the Vertical Pod Autoscaler components in cronjobs.docx?"
q_debiased = "What release manifest URL installs the Vertical Pod Autoscaler components?"

def analyze_q14(query: str, label: str):
    out.append(f"\n--- QUERY: {label} ---")
    out.append(f"Text: '{query}'")
    
    dense_vec = embed_query(query)
    sparse_vec = embed_sparse_query(query)
    s_idx = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
    s_val = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

    # 1. Dense Only Top-15
    dense_pts = client.query_points(
        collection_name=COLLECTION,
        query=dense_vec,
        using="dense",
        limit=15,
        with_payload=True
    ).points

    # 2. Native Hybrid Top-15
    hybrid_pts = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=15),
            models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15,
        with_payload=True
    ).points

    # Dense Reranking
    dense_passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(dense_pts)]
    dense_reranked = ranker.rerank(RerankRequest(query=query, passages=dense_passages)) if dense_passages else []
    dense_rerank_map = {r["id"]: (rank, r["score"]) for rank, r in enumerate(dense_reranked, 1)}

    # Hybrid Reranking
    hybrid_passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(hybrid_pts)]
    hybrid_reranked = ranker.rerank(RerankRequest(query=query, passages=hybrid_passages)) if hybrid_passages else []
    hybrid_rerank_map = {r["id"]: (rank, r["score"]) for rank, r in enumerate(hybrid_reranked, 1)}

    out.append("\n  [DENSE ONLY TOP-15 PRE-RERANK & POST-RERANK]")
    cron_found_dense = False
    for rank, p in enumerate(dense_pts, 1):
        src = p.payload.get("source", "Unknown")
        sec = p.payload.get("section", "General")
        is_target = "cronjobs.docx" in src and "vpa-release" in p.payload.get("text", "")
        post_rank, post_score = dense_rerank_map.get(rank - 1, ("-", 0.0))
        marker = " <=== TARGET CHUNK" if is_target else ""
        out.append(f"    Pre-rank {rank:2d} (Qdrant score: {p.score:.4f}) | Post-rank {post_rank} (Rerank score: {post_score:.4f}) | {src} -> {sec}{marker}")
        if is_target:
            cron_found_dense = True

    if not cron_found_dense:
        out.append("    Target cronjobs.docx chunk was NOT in Dense top-15 pre-rerank.")

    out.append("\n  [HYBRID TOP-15 PRE-RERANK & POST-RERANK]")
    cron_found_hybrid = False
    for rank, p in enumerate(hybrid_pts, 1):
        src = p.payload.get("source", "Unknown")
        sec = p.payload.get("section", "General")
        is_target = "cronjobs.docx" in src and "vpa-release" in p.payload.get("text", "")
        post_rank, post_score = hybrid_rerank_map.get(rank - 1, ("-", 0.0))
        marker = " <=== TARGET CHUNK" if is_target else ""
        out.append(f"    Pre-rank {rank:2d} (RRF score: {p.score:.4f}) | Post-rank {post_rank} (Rerank score: {post_score:.4f}) | {src} -> {sec}{marker}")
        if is_target:
            cron_found_hybrid = True

    if not cron_found_hybrid:
        out.append("    Target cronjobs.docx chunk was NOT in Hybrid top-15 pre-rerank.")

analyze_q14(q_original, "ORIGINAL (with filename 'in cronjobs.docx')")
analyze_q14(q_debiased, "DE-BIASED (filename removed)")

out.append("\n" + "="*80)
out.append("GREP CHECK: CONFIRMING ZERO FILENAMES IN ALL QUESTION TEXTS")
out.append("="*80)

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)["rag_samples"]
with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
    hard_data = json.load(f)

pattern = re.compile(r"\.(docx|html|pptx|pdf|txt)", re.IGNORECASE)
golden_violations = [(q["id"], q["question"]) for q in golden_data if pattern.search(q["question"])]
hard_violations = [(q["id"], q["question"]) for q in hard_data if pattern.search(q["question"])]

out.append(f"Total Golden Questions: {len(golden_data)} | Questions containing file extensions: {len(golden_violations)}")
if golden_violations:
    for vid, vq in golden_violations:
        out.append(f"  VIOLATION: Golden Q{vid}: {vq}")
else:
    out.append("  PASS: 0 filenames found in evals/golden_dataset.json questions.")

out.append(f"Total Hard Questions:   {len(hard_data)} | Questions containing file extensions: {len(hard_violations)}")
if hard_violations:
    for vid, vq in hard_violations:
        out.append(f"  VIOLATION: Hard Q{vid}: {vq}")
else:
    out.append("  PASS: 0 filenames found in evals/data/hard_retrieval_22.json questions.")

out.append("\n" + "="*80)
out.append("RE-RUNNING RETRIEVAL A/B BENCHMARK (scripts/run_retrieval_ab.py)")
out.append("="*80)

res_ab = subprocess.run([sys.executable, "scripts/run_retrieval_ab.py"], capture_output=True, text=True, encoding="utf-8")
out.append(res_ab.stdout)
if res_ab.stderr:
    out.append("\n[STDERR]:\n" + res_ab.stderr)

full_output = "\n".join(out) + "\n"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(full_output)
