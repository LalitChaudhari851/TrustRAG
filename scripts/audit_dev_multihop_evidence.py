"""
TrustRAG Baseline Audit: Trace 6 Dev Multi-Hop Queries.

Traces each required evidence chunk through:
  1. Raw Qdrant Hybrid RRF retrieval (ranks 1 to 15)
  2. FlashRank cross-encoder scoring
  3. Absolute floor threshold (0.08)
  4. Relative margin cutoff (top_score * 0.80)
  5. Top-5 reranked cap

Pinpoints exactly where required evidence is lost for each multi-hop case.
"""

from __future__ import annotations

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from flashrank import Ranker, RerankRequest
from evals.retrieval_dataset_schema import load_dataset


def run_multihop_audit():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    dev_multihop = [r for r in records if r.split == "dev" and r.query_type == "answerable_multi_hop"]

    ranker = Ranker()
    floor = settings.RERANKER_MIN_SCORE
    margin = settings.RERANKER_RELATIVE_MARGIN

    print(f"=== AUDITING 6 DEV MULTI-HOP QUERIES (floor={floor}, margin={margin}) ===\n")
    audit_summary = []

    for r in dev_multihop:
        print(f"================================================================================")
        print(f"Query ID:   {r.id}")
        print(f"Query:      {r.query}")
        print(f"Candidates: {r.candidate_chunk_ids}")
        print(f"Sufficient: {r.sufficient_evidence_sets}")
        print(f"Quotes:     {r.evidence_quotes}")
        print("--------------------------------------------------------------------------------")

        # 1. Search full corpus
        raw_res = search_enterprise_knowledge(r.query, limit=15, hybrid=True)
        raw_ids = [d["id"] for d in raw_res]

        print("Stage 1: Raw Qdrant Hybrid RRF Top-15:")
        for rank, doc in enumerate(raw_res, start=1):
            is_rel = doc["id"] in r.candidate_chunk_ids
            mark = " <-- [REQUIRED CHUNK]" if is_rel else ""
            print(f"  Rank {rank:2d}: ID={doc['id']} (Source: {doc.get('source')}:{doc.get('chunk_index')}){mark}")

        raw_hits = set(raw_ids[:5]) & set(r.candidate_chunk_ids)
        raw_15_hits = set(raw_ids) & set(r.candidate_chunk_ids)
        print(f"\n  Summary Stage 1: Hits in Top-5={len(raw_hits)}/{len(r.candidate_chunk_ids)} | Hits in Top-15={len(raw_15_hits)}/{len(r.candidate_chunk_ids)}")

        # 2. FlashRank Scoring
        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(raw_res)]
        req = RerankRequest(query=r.query, passages=passages)
        f_res = ranker.rerank(req)

        top_score = float(f_res[0]["score"]) if f_res else 0.0
        rel_cutoff = max(floor, top_score * (1.0 - margin))
        print(f"\nStage 2: FlashRank Reranker (Top Score: {top_score:.4f} | Margin Cutoff: {rel_cutoff:.4f} | Floor: {floor:.4f}):")

        kept = []
        dropped_margin = []
        dropped_floor = []

        for rank, res in enumerate(f_res, start=1):
            orig_doc = raw_res[int(res["id"])]
            score = float(res["score"])
            doc_id = orig_doc["id"]
            is_rel = doc_id in r.candidate_chunk_ids
            mark = " <-- [REQUIRED CHUNK]" if is_rel else ""

            status = "KEPT"
            if score < floor:
                status = "DROPPED_FLOOR"
                dropped_floor.append((doc_id, score, is_rel))
            elif score < rel_cutoff:
                status = "DROPPED_MARGIN"
                dropped_margin.append((doc_id, score, is_rel))
            else:
                if len(kept) < 5:
                    kept.append((doc_id, score, is_rel))
                else:
                    status = "DROPPED_TOP5_CAP"

            print(f"  Rerank #{rank:2d}: Score={score:.4f} | Action={status:<17} | ID={doc_id} ({orig_doc.get('source')}:{orig_doc.get('chunk_index')}){mark}")

        rerank_hits = [k[0] for k in kept if k[2]]
        print(f"\nStage 2 Final Kept Chunks in Top-5 ({len(kept)} chunks):")
        for k in kept:
            print(f"  - ID={k[0]} (Score={k[1]:.4f}, Relevant={k[2]})")

        # Conjunction checks
        kept_id_set = set(k[0] for k in kept)
        conj_success = any(set(s).issubset(kept_id_set) for s in r.sufficient_evidence_sets)
        raw_conj_success = any(set(s).issubset(set(raw_ids[:5])) for s in r.sufficient_evidence_sets)

        print(f"\nConjunction Results:")
        print(f"  Raw Top-5 Conjunction Success:      {raw_conj_success}")
        print(f"  Reranked Top-5 Conjunction Success: {conj_success}")

        # Diagnosis
        loss_reasons = []
        for cid in r.candidate_chunk_ids:
            if cid not in raw_ids:
                loss_reasons.append(f"Chunk {cid} completely missed in Qdrant first-stage top-15")
            elif cid in [df[0] for df in dropped_floor]:
                score = next(df[1] for df in dropped_floor if df[0] == cid)
                loss_reasons.append(f"Chunk {cid} retrieved at rank {raw_ids.index(cid)+1} but dropped by absolute floor ({score:.4f} < {floor:.4f})")
            elif cid in [dm[0] for dm in dropped_margin]:
                score = next(dm[1] for dm in dropped_margin if dm[0] == cid)
                loss_reasons.append(f"Chunk {cid} retrieved at rank {raw_ids.index(cid)+1} but dropped by relative margin ({score:.4f} < {rel_cutoff:.4f}, top was {top_score:.4f})")
            elif cid in [k[0] for k in kept]:
                loss_reasons.append(f"Chunk {cid} preserved in reranked top-5 (Score: {next(k[1] for k in kept if k[0] == cid):.4f})")
            else:
                loss_reasons.append(f"Chunk {cid} dropped by top-5 capacity limit")

        print("Diagnosis of Evidence Retention / Loss:")
        for lr in loss_reasons:
            print(f"  * {lr}")
        print("\n")

        audit_summary.append({
            "query_id": r.id,
            "query": r.query,
            "candidate_count": len(r.candidate_chunk_ids),
            "raw_top5_hits": len(raw_hits),
            "raw_top15_hits": len(raw_15_hits),
            "rerank_top5_hits": len(rerank_hits),
            "raw_conj_success": raw_conj_success,
            "rerank_conj_success": conj_success,
            "loss_reasons": loss_reasons,
        })

    return audit_summary


if __name__ == "__main__":
    run_multihop_audit()
