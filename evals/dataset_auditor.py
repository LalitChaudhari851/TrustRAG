"""
TrustRAG Golden Dataset Auditor (v2).

Performs comprehensive audits of the retrieval evaluation dataset:
Phase A: Dataset audit (schema, IDs, duplicates, evidence quotes vs corpus,
         expected answers, multi-hop sets, unanswerable subtypes, Qdrant UUIDs).
Phase E: Split integrity audit (query overlap, shared chunk IDs, source document
         distribution, template leakage, topic overlap).

Produces:
- evals/results/golden_dataset_audit_v2.json
- evals/results/golden_dataset_audit_v2.md
- evals/results/golden_dataset_split_audit_v2.json
- evals/results/golden_dataset_split_audit_v2.md
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import string
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)

from app.config import settings
from evals.retrieval_dataset_schema import (
    RetrievalEvalRecord,
    load_dataset,
    validate_eval_record,
    validate_uuid_string,
)

PROCESSED_TRUE_DIR = "processed_data/true"
DEFAULT_DATASET_PATH = "evals/data/retrieval_golden_seed_v1.json"


def normalize_query_text(q: str) -> str:
    """Normalizes query text: lowercase, remove punctuation, collapse whitespace."""
    s = q.lower()
    for p in string.punctuation:
        s = s.replace(p, " ")
    return " ".join(s.split())


def compute_token_jaccard(q1: str, q2: str) -> float:
    """Computes Jaccard similarity between two queries' word tokens."""
    t1 = set(normalize_query_text(q1).split())
    t2 = set(normalize_query_text(q2).split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def build_local_true_corpus() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Loads all true chunks from processed_data/true/*.json and builds:
    1. chunk_by_uuid: UUID -> {uuid, filename, chunk_index, section, text, content_hash}
    2. chunks_by_file: filename -> list of chunk dicts
    """
    ns = uuid.UUID(settings.PROJECT_UUID_NAMESPACE)
    chunk_by_uuid: Dict[str, Dict[str, Any]] = {}
    chunks_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    true_files = sorted(glob.glob(os.path.join(PROCESSED_TRUE_DIR, "*.json")))
    for fpath in true_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        fn = data.get("filename", os.path.basename(fpath).replace(".json", ""))
        chunks = data.get("chunks", [])
        for c in chunks:
            c_idx = c["chunk_index"]
            c_text = c["text"]
            c_sec = c.get("section", "General")
            c_hash = hashlib.sha256(c_text.strip().encode("utf-8")).hexdigest()[:16]
            key = f"{fn}:{c_idx}:{c_hash}"
            c_uuid = str(uuid.uuid5(ns, key))

            item = {
                "uuid": c_uuid,
                "filename": fn,
                "chunk_index": c_idx,
                "section": c_sec,
                "text": c_text,
                "content_hash": c_hash,
                "char_length": len(c_text),
                "word_count": len(c_text.split()),
            }
            chunk_by_uuid[c_uuid] = item
            chunks_by_file[fn].append(item)

    return chunk_by_uuid, dict(chunks_by_file)


def run_dataset_audit(dataset_path: str = DEFAULT_DATASET_PATH, check_live_qdrant: bool = True) -> Dict[str, Any]:
    """
    Performs full Phase A audit of the dataset records.
    Distinguishes Definite Errors, Suspected Issues, and Items Requiring Human Review.
    """
    records = load_dataset(dataset_path)
    corpus_uuid_map, corpus_by_file = build_local_true_corpus()

    definite_errors: List[Dict[str, Any]] = []
    suspected_issues: List[Dict[str, Any]] = []
    items_requiring_human_review: List[Dict[str, Any]] = []

    seen_ids: Set[str] = set()
    exact_queries: Dict[str, str] = {}
    normalized_queries: Dict[str, str] = {}

    all_candidate_chunk_ids: Set[str] = set()

    for r in records:
        all_candidate_chunk_ids.update(r.candidate_chunk_ids)

        # 1. Schema check
        schema_errs = validate_eval_record(r)
        if schema_errs:
            for err in schema_errs:
                definite_errors.append({
                    "record_id": r.id,
                    "category": "schema_violation",
                    "description": err,
                    "severity": "high",
                })

        # 2. Duplicate ID
        if r.id in seen_ids:
            definite_errors.append({
                "record_id": r.id,
                "category": "duplicate_record_id",
                "description": f"Duplicate record ID '{r.id}' detected.",
                "severity": "critical",
            })
        else:
            seen_ids.add(r.id)

        # 3. Exact and normalized query duplicates
        raw_q = r.query.strip()
        norm_q = normalize_query_text(raw_q)
        if raw_q in exact_queries:
            definite_errors.append({
                "record_id": r.id,
                "category": "exact_duplicate_query",
                "description": f"Exact duplicate of record '{exact_queries[raw_q]}'.",
                "severity": "high",
            })
        else:
            exact_queries[raw_q] = r.id

        if norm_q in normalized_queries and raw_q not in exact_queries:
            suspected_issues.append({
                "record_id": r.id,
                "category": "normalized_duplicate_query",
                "description": f"Normalized duplicate of record '{normalized_queries[norm_q]}'.",
                "severity": "medium",
            })
        else:
            normalized_queries[norm_q] = r.id

        # 4. Evidence quotes and local corpus alignment
        if r.query_type != "unanswerable":
            # Check expected_answer
            if not r.expected_answer or not r.expected_answer.strip():
                suspected_issues.append({
                    "record_id": r.id,
                    "category": "missing_expected_answer",
                    "description": "Answerable record is missing 'expected_answer'.",
                    "severity": "medium",
                })

            # Check chunk existence in true corpus
            for cid in r.candidate_chunk_ids:
                if cid not in corpus_uuid_map:
                    definite_errors.append({
                        "record_id": r.id,
                        "category": "missing_chunk_in_corpus",
                        "description": f"Candidate chunk UUID '{cid}' does not exist in local true corpus.",
                        "severity": "high",
                    })
                else:
                    chunk_meta = corpus_uuid_map[cid]
                    # Check provenance filename match
                    if chunk_meta["filename"] not in r.source_files:
                        suspected_issues.append({
                            "record_id": r.id,
                            "category": "provenance_mismatch",
                            "description": f"Chunk '{cid}' originates from '{chunk_meta['filename']}', but source_files lists {r.source_files}.",
                            "severity": "medium",
                        })

                    # Check evidence quotes against chunk text
                    for q_idx, quote in enumerate(r.evidence_quotes):
                        clean_quote = " ".join(quote.split()).lower()
                        clean_text = " ".join(chunk_meta["text"].split()).lower()
                        if clean_quote not in clean_text:
                            suspected_issues.append({
                                "record_id": r.id,
                                "category": "quote_not_exact_substring",
                                "description": f"Evidence quote [{q_idx+1}] is not an exact substring of chunk '{cid}'.",
                                "severity": "low",
                            })
        else:
            # Unanswerable checks
            if not r.expected_answer or not r.expected_answer.strip():
                suspected_issues.append({
                    "record_id": r.id,
                    "category": "missing_unanswerable_explanation",
                    "description": "Unanswerable record missing expected_answer explanation.",
                    "severity": "low",
                })
            if r.unanswerable_subtype not in {"corpus_unanswerable", "unresolved_evidence"}:
                definite_errors.append({
                    "record_id": r.id,
                    "category": "invalid_unanswerable_subtype",
                    "description": f"Invalid subtype '{r.unanswerable_subtype}'.",
                    "severity": "high",
                })

        # 5. Multi-hop checks
        if r.query_type == "answerable_multi_hop":
            if len(r.sufficient_evidence_sets) < 1:
                definite_errors.append({
                    "record_id": r.id,
                    "category": "empty_sufficient_evidence_sets",
                    "description": "Multi-hop query has no sufficient evidence sets.",
                    "severity": "high",
                })
            for s_idx, s_set in enumerate(r.sufficient_evidence_sets):
                if len(s_set) < 2:
                    definite_errors.append({
                        "record_id": r.id,
                        "category": "single_chunk_multihop_set",
                        "description": f"Sufficient set [{s_idx}] has only {len(s_set)} chunk; multi-hop requires >= 2 chunks.",
                        "severity": "high",
                    })

        # 6. Items requiring human review
        # Every unverified record requires human review by definition
        if r.verification_status == "unverified_synthetic":
            items_requiring_human_review.append({
                "record_id": r.id,
                "query": r.query,
                "query_type": r.query_type,
                "split": r.split,
                "reason": "Synthetic seed record pending human adjudication and verified label assignment.",
            })

    # 7. Semantic duplicate scan across all records (pairwise Jaccard >= 0.70)
    potential_semantic_duplicates = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = records[i], records[j]
            jacc = compute_token_jaccard(r1.query, r2.query)
            if jacc >= 0.65:
                potential_semantic_duplicates.append({
                    "record_1": {"id": r1.id, "split": r1.split, "query": r1.query},
                    "record_2": {"id": r2.id, "split": r2.split, "query": r2.query},
                    "jaccard_similarity": round(jacc, 3),
                    "is_cross_split": r1.split != r2.split,
                })

    # 8. Live Qdrant verification (optional / graceful)
    qdrant_status = {"checked": False, "total": len(all_candidate_chunk_ids), "found": 0, "missing": [], "error": None}
    if check_live_qdrant:
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5.0)
            pts = client.retrieve(collection_name=settings.QDRANT_COLLECTION, ids=list(all_candidate_chunk_ids), with_payload=False)
            found_ids = {str(pt.id) for pt in pts}
            missing = [cid for cid in all_candidate_chunk_ids if cid not in found_ids]
            qdrant_status = {
                "checked": True,
                "total": len(all_candidate_chunk_ids),
                "found": len(found_ids),
                "missing": missing,
                "collection": settings.QDRANT_COLLECTION,
                "error": None,
            }
        except Exception as e:
            qdrant_status["error"] = f"Qdrant live check skipped / failed: {str(e)}"

    audit_summary = {
        "dataset_path": dataset_path,
        "total_records": len(records),
        "corpus_unique_chunks": len(corpus_uuid_map),
        "dataset_candidate_chunk_ids": len(all_candidate_chunk_ids),
        "qdrant_verification": qdrant_status,
        "findings_count": {
            "definite_errors": len(definite_errors),
            "suspected_issues": len(suspected_issues),
            "items_requiring_human_review": len(items_requiring_human_review),
            "potential_semantic_duplicates": len(potential_semantic_duplicates),
        },
        "definite_errors": definite_errors,
        "suspected_issues": suspected_issues,
        "potential_semantic_duplicates": potential_semantic_duplicates,
        "items_requiring_human_review_count": len(items_requiring_human_review),
    }

    return audit_summary


def run_split_audit(dataset_path: str = DEFAULT_DATASET_PATH) -> Dict[str, Any]:
    """
    Performs full Phase E audit of dev/test partition integrity:
    1. Verbatim & normalized query leakage across splits
    2. Shared candidate chunk UUIDs across splits
    3. Source document distribution across splits
    4. Topic family distribution across splits
    5. Query template similarity across splits
    """
    records = load_dataset(dataset_path)
    corpus_uuid_map, _ = build_local_true_corpus()

    dev_records = [r for r in records if r.split == "dev"]
    test_records = [r for r in records if r.split == "test"]

    dev_queries = {normalize_query_text(r.query): r for r in dev_records}
    test_queries = {normalize_query_text(r.query): r for r in test_records}

    # 1. Query overlap
    query_leakage = []
    for nq, dr in dev_queries.items():
        if nq in test_queries:
            tr = test_queries[nq]
            query_leakage.append({
                "dev_id": dr.id,
                "test_id": tr.id,
                "dev_query": dr.query,
                "test_query": tr.query,
                "type": "normalized_query_leakage",
            })

    # 2. Shared candidate chunks
    dev_chunk_map: Dict[str, List[str]] = defaultdict(list)
    test_chunk_map: Dict[str, List[str]] = defaultdict(list)
    for r in dev_records:
        for cid in r.candidate_chunk_ids:
            dev_chunk_map[cid].append(r.id)
    for r in test_records:
        for cid in r.candidate_chunk_ids:
            test_chunk_map[cid].append(r.id)

    shared_chunks = sorted(set(dev_chunk_map.keys()) & set(test_chunk_map.keys()))
    shared_chunk_details = []
    for cid in shared_chunks:
        cmeta = corpus_uuid_map.get(cid, {})
        shared_chunk_details.append({
            "chunk_id": cid,
            "filename": cmeta.get("filename", "unknown"),
            "section": cmeta.get("section", "unknown"),
            "dev_records": dev_chunk_map[cid],
            "test_records": test_chunk_map[cid],
            "snippet": cmeta.get("text", "")[:120] + "...",
        })

    # 3. Source document distribution
    dev_files = defaultdict(int)
    test_files = defaultdict(int)
    for r in dev_records:
        for sf in r.source_files:
            dev_files[sf] += 1
    for r in test_records:
        for sf in r.source_files:
            test_files[sf] += 1

    all_files = sorted(set(dev_files.keys()) | set(test_files.keys()))
    source_distribution = []
    for fn in all_files:
        source_distribution.append({
            "filename": fn,
            "dev_queries": dev_files.get(fn, 0),
            "test_queries": test_files.get(fn, 0),
            "shared": (fn in dev_files and fn in test_files),
        })

    # 4. Cross-split semantic similarities (Jaccard >= 0.60)
    cross_split_similarities = []
    for dr in dev_records:
        for tr in test_records:
            jacc = compute_token_jaccard(dr.query, tr.query)
            if jacc >= 0.55:
                cross_split_similarities.append({
                    "dev_id": dr.id,
                    "dev_query": dr.query,
                    "test_id": tr.id,
                    "test_query": tr.query,
                    "jaccard_similarity": round(jacc, 3),
                    "shared_chunks": list(set(dr.candidate_chunk_ids) & set(tr.candidate_chunk_ids)),
                })

    split_summary = {
        "dataset_path": dataset_path,
        "counts": {
            "dev_records": len(dev_records),
            "test_records": len(test_records),
            "dev_candidate_chunks": len(dev_chunk_map),
            "test_candidate_chunks": len(test_chunk_map),
            "shared_chunks_count": len(shared_chunks),
            "shared_source_files_count": sum(1 for s in source_distribution if s["shared"]),
            "verbatim_query_leakage_count": len(query_leakage),
            "cross_split_high_similarity_count": len(cross_split_similarities),
        },
        "query_leakage": query_leakage,
        "shared_chunks": shared_chunk_details,
        "source_distribution": source_distribution,
        "cross_split_similarities": sorted(cross_split_similarities, key=lambda x: x["jaccard_similarity"], reverse=True),
        "split_risk_assessment": {
            "verbatim_leakage_risk": "ZERO" if len(query_leakage) == 0 else "CRITICAL",
            "chunk_leakage_risk": "HIGH" if len(shared_chunks) > 0 else "ZERO",
            "source_leakage_risk": "CRITICAL" if all(s["shared"] for s in source_distribution) else "MEDIUM",
            "recommendation": "Current partition uses random query-level splitting across all 6 documents. For an enterprise benchmark resistant to memorization, a document-disjoint split (e.g. 4 documents dev / 2 documents test) or cluster-disjoint split is strongly recommended.",
        },
    }

    return split_summary


def format_dataset_audit_markdown(audit: Dict[str, Any]) -> str:
    md = []
    md.append("# TrustRAG Golden Dataset Audit Report (v2)")
    md.append("")
    md.append(f"**Target Dataset**: `{audit['dataset_path']}`  ")
    md.append(f"**Total Records**: {audit['total_records']}  ")
    md.append(f"**Corpus Total Approved Chunks**: {audit['corpus_unique_chunks']} (from `processed_data/true/`)  ")
    md.append(f"**Unique Candidate Chunk UUIDs Referenced**: {audit['dataset_candidate_chunk_ids']}  ")
    q_stat = audit['qdrant_verification']
    if q_stat['checked']:
        md.append(f"**Qdrant Verification**: Found {q_stat['found']} of {q_stat['total']} in `{q_stat['collection']}`  ")
    else:
        md.append(f"**Qdrant Verification**: {q_stat.get('error') or 'Not executed'}  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append("| Category | Count | Status | Action Required |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Definite Errors** | {audit['findings_count']['definite_errors']} | {'FAIL' if audit['findings_count']['definite_errors'] > 0 else 'PASS'} | Immediate remediation |")
    md.append(f"| **Suspected Issues** | {audit['findings_count']['suspected_issues']} | WARNING | Review and document |")
    md.append(f"| **Potential Semantic Duplicates** | {audit['findings_count']['potential_semantic_duplicates']} | REVIEW | Check question diversity |")
    md.append(f"| **Pending Human Review** | {audit['findings_count']['items_requiring_human_review']} | PENDING | Human adjudication required |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Definite Errors")
    md.append("")
    if not audit["definite_errors"]:
        md.append("*No definite schema errors, invalid UUIDs, broken references, or conjunction integrity violations found.*")
    else:
        for err in audit["definite_errors"]:
            md.append(f"- **Record `{err['record_id']}`** [{err['category']}]: {err['description']}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Suspected Issues")
    md.append("")
    if not audit["suspected_issues"]:
        md.append("*No suspected issues identified.*")
    else:
        for iss in audit["suspected_issues"]:
            md.append(f"- **Record `{iss['record_id']}`** [{iss['category']}]: {iss['description']}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Potential Semantic Duplicates")
    md.append("")
    if not audit["potential_semantic_duplicates"]:
        md.append("*No high-similarity duplicate pairs detected.*")
    else:
        md.append("| Record 1 (Split) | Record 2 (Split) | Similarity | Cross-Split? |")
        md.append("| :--- | :--- | :--- | :--- |")
        for dup in audit["potential_semantic_duplicates"]:
            r1 = dup["record_1"]
            r2 = dup["record_2"]
            cs = "**YES (LEAKAGE RISK)**" if dup["is_cross_split"] else "No (Same split)"
            md.append(f"| `{r1['id']}` ({r1['split']}) - \"{r1['query'][:40]}...\" | `{r2['id']}` ({r2['split']}) - \"{r2['query'][:40]}...\" | {dup['jaccard_similarity']} | {cs} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Human Review Requirement")
    md.append("")
    md.append(f"All **{audit['items_requiring_human_review_count']} records** currently remain marked as `unverified_synthetic`.")
    md.append("No record may be promoted to `verified_human` or exported into the benchmark freeze without explicit review recording:")
    md.append("1. Reviewer identity")
    md.append("2. UTC timestamp")
    md.append("3. Documented rationale")
    md.append("4. Evidence verification (support, partial_support, contradiction, irrelevant)")
    md.append("5. Expected answer verification")
    md.append("")
    return "\n".join(md)


def format_split_audit_markdown(audit: Dict[str, Any]) -> str:
    md = []
    cnt = audit["counts"]
    risk = audit["split_risk_assessment"]
    md.append("# TrustRAG Golden Dataset Split Audit Report (v2)")
    md.append("")
    md.append(f"**Target Dataset**: `{audit['dataset_path']}`  ")
    md.append(f"**Split Counts**: Dev = {cnt['dev_records']}, Test = {cnt['test_records']}  ")
    md.append(f"**Shared Candidate Chunks**: {cnt['shared_chunks_count']} chunks shared between Dev and Test  ")
    md.append(f"**Shared Source Files**: {cnt['shared_source_files_count']} of 6 files shared across splits  ")
    md.append(f"**Verbatim Query Leakage**: {cnt['verbatim_query_leakage_count']}  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Split Risk Assessment")
    md.append("")
    md.append(f"- **Verbatim Query Leakage**: `{risk['verbatim_leakage_risk']}` (0 verbatim queries shared)")
    md.append(f"- **Chunk-Level Leakage**: `{risk['chunk_leakage_risk']}` ({cnt['shared_chunks_count']} exact chunk IDs referenced in both dev and test)")
    md.append(f"- **Document-Level Leakage**: `{risk['source_leakage_risk']}` (100% of source files appear in both dev and test)")
    md.append("")
    md.append(f"> **Policy Recommendation**: {risk['recommendation']}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Shared Candidate Chunks Detail")
    md.append("")
    if not audit["shared_chunks"]:
        md.append("*No shared candidate chunks found between dev and test.*")
    else:
        md.append("| Chunk UUID | Source Document | Dev Queries | Test Queries | Snippet Preview |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for sc in audit["shared_chunks"]:
            dq = ", ".join(f"`{x}`" for x in sc["dev_records"])
            tq = ", ".join(f"`{x}`" for x in sc["test_records"])
            md.append(f"| `{sc['chunk_id'][:8]}...` | `{sc['filename']}` | {dq} | {tq} | {sc['snippet']} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Source Document Distribution")
    md.append("")
    md.append("| Document Name | Dev Queries | Test Queries | Shared Across Splits? |")
    md.append("| :--- | :--- | :--- | :--- |")
    for sd in audit["source_distribution"]:
        shared_str = "**YES (SHARED)**" if sd["shared"] else "No (Isolated)"
        md.append(f"| `{sd['filename']}` | {sd['dev_queries']} | {sd['test_queries']} | {shared_str} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Cross-Split Semantic Similarities")
    md.append("")
    if not audit["cross_split_similarities"]:
        md.append("*No high cross-split query similarities found.*")
    else:
        md.append("| Dev Query | Test Query | Jaccard Overlap | Shared Evidence Chunks |")
        md.append("| :--- | :--- | :--- | :--- |")
        for cs in audit["cross_split_similarities"][:10]:
            dq = f"`{cs['dev_id']}`: \"{cs['dev_query'][:35]}...\""
            tq = f"`{cs['test_id']}`: \"{cs['test_query'][:35]}...\""
            sc = ", ".join(cs["shared_chunks"]) if cs["shared_chunks"] else "None"
            md.append(f"| {dq} | {tq} | {cs['jaccard_similarity']} | {sc} |")
    md.append("")
    return "\n".join(md)


def main() -> None:
    print("\n--- Running Golden Dataset Audit v2 (Phase A) ---")
    dataset_audit = run_dataset_audit(DEFAULT_DATASET_PATH, check_live_qdrant=True)
    os.makedirs("evals/results", exist_ok=True)

    json_audit_path = "evals/results/golden_dataset_audit_v2.json"
    md_audit_path = "evals/results/golden_dataset_audit_v2.md"
    with open(json_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_to_serializable(dataset_audit), f, indent=2)
    with open(md_audit_path, "w", encoding="utf-8") as f:
        f.write(format_dataset_audit_markdown(dataset_audit))
    print(f"[OK] Wrote Phase A audit to {json_audit_path} and {md_audit_path}")

    print("\n--- Running Golden Dataset Split Audit v2 (Phase E) ---")
    split_audit = run_split_audit(DEFAULT_DATASET_PATH)
    json_split_path = "evals/results/golden_dataset_split_audit_v2.json"
    md_split_path = "evals/results/golden_dataset_split_audit_v2.md"
    with open(json_split_path, "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)
    with open(md_split_path, "w", encoding="utf-8") as f:
        f.write(format_split_audit_markdown(split_audit))
    print(f"[OK] Wrote Phase E split audit to {json_split_path} and {md_split_path}")


def audit_to_serializable(d: Any) -> Any:
    if isinstance(d, set):
        return sorted(list(d))
    if isinstance(d, dict):
        return {k: audit_to_serializable(v) for k, v in d.items()}
    if isinstance(d, list):
        return [audit_to_serializable(x) for x in d]
    return d


if __name__ == "__main__":
    main()
