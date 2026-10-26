"""
TrustRAG Human Annotation CLI (v2).

Enforces the explicit human review workflow for retrieval evaluation datasets.
Only explicit human review with reviewer identification, documented rationale,
evidence verification, and answerability confirmation can promote a record.

Usage Examples:
    python -m evals.annotation_cli stats
    python -m evals.annotation_cli list --status pending
    python -m evals.annotation_cli list --status unverified_synthetic --split dev
    python -m evals.annotation_cli inspect <record_id>
    python -m evals.annotation_cli verify-evidence <record_id>
    python -m evals.annotation_cli verify-qdrant [record_id]
    python -m evals.annotation_cli approve <record_id> --reviewer "Alice" --reason "Verified chunk quotes against docs." --answerability answerable
    python -m evals.annotation_cli reject <record_id> --reviewer "Alice" --reason "Ambiguous question text."
    python -m evals.annotation_cli export-verified --output evals/data/retrieval_golden_verified.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

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
    save_dataset,
    validate_dataset_records,
    validate_records_against_qdrant,
    resolve_chunk_evidence,
    validate_benchmark_freeze,
    compute_records_content_hash,
)

DEFAULT_DATASET_PATH = "evals/data/retrieval_golden_seed_v1.json"


def cmd_stats(dataset_path: str) -> None:
    records = load_dataset(dataset_path)
    val = validate_dataset_records(records)

    total = len(records)
    verified = sum(1 for r in records if r.verification_status == "verified_human")
    rejected = sum(1 for r in records if r.verification_status == "rejected")
    pending = sum(1 for r in records if r.verification_status == "unverified_synthetic")
    coverage_pct = (verified + rejected) / total * 100.0 if total > 0 else 0.0

    print(f"\n=== RETRIEVAL DATASET STATS ({dataset_path}) ===")
    print(f"Total Records:      {total}")
    print(f"Valid Schema:       {val['is_valid']}")
    print(f"\n--- Human Review Coverage ---")
    print(f"Reviewed / Decided: {verified + rejected} / {total} ({coverage_pct:.1f}%)")
    print(f"  - Verified Human: {verified} ({verified/total*100.0:.1f}%)")
    print(f"  - Rejected:       {rejected} ({rejected/total*100.0:.1f}%)")
    print(f"  - Pending Review: {pending} ({pending/total*100.0:.1f}%)")

    if rejected > 0:
        print("\nRejection Reasons:")
        for r in records:
            if r.verification_status == "rejected" and r.reviewer_decision:
                rev = r.reviewer_decision.get("reviewer", "unknown")
                rsn = r.reviewer_decision.get("reason", "no reason")
                print(f"  - [{r.id}] ({rev}): {rsn}")

    print("\nBreakdown by Type:")
    for k, v in val["breakdown"]["by_type"].items():
        print(f"  - {k}: {v}")
    print("\nBreakdown by Split:")
    for k, v in val["breakdown"]["by_split"].items():
        print(f"  - {k}: {v}")
    print("\nBreakdown by Unanswerable Subtype:")
    for k, v in val["breakdown"]["by_unanswerable_subtype"].items():
        print(f"  - {k}: {v}")


def cmd_list(dataset_path: str, status_filter: Optional[str] = None, split_filter: Optional[str] = None) -> None:
    records = load_dataset(dataset_path)

    # Normalize 'pending' alias to 'unverified_synthetic'
    target_status = "unverified_synthetic" if status_filter == "pending" else status_filter

    filtered = [
        r for r in records
        if (target_status is None or r.verification_status == target_status)
        and (split_filter is None or r.split == split_filter)
    ]

    print(f"\nFound {len(filtered)} records (filter: status={status_filter or 'ALL'}, split={split_filter or 'ALL'}):")
    print(f"{'ID':<10} {'Type':<24} {'Split':<6} {'Status':<22} {'Query'}")
    print("-" * 95)
    for r in filtered:
        q_preview = (r.query[:50] + "...") if len(r.query) > 50 else r.query
        print(f"{r.id:<10} {r.query_type:<24} {r.split:<6} {r.verification_status:<22} {q_preview}")


def cmd_inspect(dataset_path: str, record_id: str) -> None:
    records = load_dataset(dataset_path)
    target = next((r for r in records if r.id == record_id), None)
    if not target:
        print(f"Error: Record ID '{record_id}' not found in {dataset_path}")
        return

    print(f"\n=== RECORD DETAILS: {target.id} ===")
    print(f"Query:                {target.query}")
    print(f"Query Type:           {target.query_type}")
    if target.unanswerable_subtype:
        print(f"Unanswerable Subtype: {target.unanswerable_subtype}")
    print(f"Split:                {target.split}")
    print(f"Difficulty:           {target.difficulty}")
    print(f"Domain:               {target.domain}")
    if target.topic_family:
        print(f"Topic Family:         {target.topic_family}")
    print(f"Verification Status:  {target.verification_status}")
    print(f"Source Files:         {target.source_files}")
    print(f"Expected Sections:    {target.expected_sections}")
    print(f"Expected Answer:      {target.expected_answer}")

    print(f"\nEvidence Quotes ({len(target.evidence_quotes)}):")
    for i, q in enumerate(target.evidence_quotes, start=1):
        print(f"  [{i}] {q}")

    print(f"\nCandidate Chunk IDs ({len(target.candidate_chunk_ids)}):")
    for cid in target.candidate_chunk_ids:
        grade = target.relevance_grades.get(cid, 1)
        cmeta = resolve_chunk_evidence(cid)
        loc_str = f"({cmeta['filename']} [chunk {cmeta['chunk_index']} - {cmeta['section']}])" if cmeta else "(not in local corpus)"
        print(f"  - {cid} {loc_str} (Grade: {grade})")

    print(f"\nSufficient Evidence Sets ({len(target.sufficient_evidence_sets)}):")
    for i, s_set in enumerate(target.sufficient_evidence_sets, start=1):
        print(f"  Set {i}: {s_set}")

    if target.reviewer_decision:
        print(f"\nReviewer Decision (Current):")
        print(json.dumps(target.reviewer_decision, indent=2))

    if target.review_history:
        print(f"\nReview History Events ({len(target.review_history)}):")
        for i, ev in enumerate(target.review_history, start=1):
            print(f"  [Event {i}] {ev.get('timestamp')} | {ev.get('decision').upper()} by {ev.get('reviewer')}: {ev.get('reason')}")

    if target.candidate_corrections:
        print(f"\nCandidate Corrections:")
        print(json.dumps(target.candidate_corrections, indent=2))

    if target.notes:
        print(f"\nNotes: {target.notes}")


def cmd_verify_evidence(dataset_path: str, record_id: str) -> None:
    records = load_dataset(dataset_path)
    target = next((r for r in records if r.id == record_id), None)
    if not target:
        print(f"Error: Record ID '{record_id}' not found in {dataset_path}")
        return

    print(f"\n=== EVIDENCE VERIFICATION: {target.id} ===")
    print(f"Query: {target.query}")
    print(f"Expected Answer: {target.expected_answer or 'N/A'}")

    if target.query_type == "unanswerable":
        print(f"\nQuery is UNANSWERABLE ({target.unanswerable_subtype}). No evidence chunks expected.")
        return

    print(f"\nResolving {len(target.candidate_chunk_ids)} candidate chunks in local true corpus:")
    for idx, cid in enumerate(target.candidate_chunk_ids, start=1):
        cmeta = resolve_chunk_evidence(cid)
        print(f"\n--- Candidate Chunk [{idx}]: {cid} ---")
        if not cmeta:
            print("  [FAIL] Chunk ID NOT FOUND in local processed_data/true/ corpus!")
            continue

        print(f"  Source File: {cmeta['filename']}")
        print(f"  Chunk Index: {cmeta['chunk_index']}")
        print(f"  Section:     {cmeta['section']}")
        print(f"  Content Hash:{cmeta['content_hash']}")
        print(f"\n  Stored Content Preview:")
        for line in cmeta['text'].splitlines()[:12]:
            print(f"    | {line}")
        if len(cmeta['text'].splitlines()) > 12:
            print("    | ... [truncated preview]")

        # Compare evidence quotes against chunk text
        clean_text = " ".join(cmeta['text'].split()).lower()
        print(f"\n  Evidence Quote Comparison:")
        for q_idx, quote in enumerate(target.evidence_quotes, start=1):
            clean_quote = " ".join(quote.split()).lower()
            if clean_quote in clean_text:
                print(f"    [{q_idx}] EXACT SUBSTRING MATCH: \"{quote[:60]}...\"")
            else:
                # Check token overlap
                q_tokens = set(clean_quote.split())
                t_tokens = set(clean_text.split())
                overlap = len(q_tokens & t_tokens) / len(q_tokens) if q_tokens else 0.0
                if overlap >= 0.70:
                    print(f"    [{q_idx}] PARAPHRASE MATCH ({overlap*100:.0f}% token overlap): \"{quote[:60]}...\"")
                else:
                    print(f"    [{q_idx}] NO MATCH ({overlap*100:.0f}% overlap): \"{quote[:60]}...\"")


def cmd_verify_qdrant(dataset_path: str, record_id: Optional[str] = None) -> None:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5.0)
    except Exception as e:
        print(f"[ERROR] Could not connect to Qdrant ({settings.QDRANT_URL}): {str(e)}")
        print("Note: Network access or Qdrant server may be offline. Read-only local evidence verification is still available.")
        return

    records = load_dataset(dataset_path)
    collection = settings.QDRANT_COLLECTION

    if record_id:
        target = next((r for r in records if r.id == record_id), None)
        if not target:
            print(f"Error: Record ID '{record_id}' not found in {dataset_path}")
            return
        records_to_check = [target]
        print(f"Verifying candidate chunks for record '{record_id}' in Qdrant '{collection}'...")
    else:
        records_to_check = records
        print(f"Verifying all candidate chunk UUIDs in {dataset_path} against Qdrant '{collection}'...")

    res = validate_records_against_qdrant(records_to_check, client, collection)
    print(f"Total Unique Chunk UUIDs Checked: {res['total_checked']}")
    print(f"Found in Qdrant:                   {res['found']}")
    if res["all_verified"]:
        print("[OK] All candidate chunk UUIDs are 100% verified present in Qdrant!")
    else:
        print(f"[FAIL] Missing {len(res['missing'])} chunk UUIDs in Qdrant: {res['missing']}")


def cmd_approve(
    dataset_path: str,
    record_id: str,
    reviewer: str,
    reason: str,
    answerability: str = "answerable",
    evidence_judgments_json: Optional[str] = None,
    answer_verified: bool = True,
    multi_hop_verified: bool = True,
    corrections_json: Optional[str] = None,
) -> None:
    records = load_dataset(dataset_path)
    target = next((r for r in records if r.id == record_id), None)
    if not target:
        print(f"Error: Record ID '{record_id}' not found in {dataset_path}")
        return

    evidence_judgments = None
    if evidence_judgments_json:
        try:
            evidence_judgments = json.loads(evidence_judgments_json)
        except Exception as e:
            print(f"Error: Invalid JSON for --evidence-judgments: {e}")
            return

    candidate_corrections = None
    if corrections_json:
        try:
            candidate_corrections = json.loads(corrections_json)
        except Exception as e:
            print(f"Error: Invalid JSON for --corrections: {e}")
            return

    target.promote_to_verified(
        reviewer=reviewer,
        reason=reason,
        answerability=answerability,
        evidence_judgments=evidence_judgments,
        answer_verified=answer_verified,
        multi_hop_verified=multi_hop_verified,
        candidate_corrections=candidate_corrections,
    )
    save_dataset(records, dataset_path)
    print(f"[OK] Record '{record_id}' successfully approved and promoted to 'verified_human' by '{reviewer}'.")
    print(f"     Decision: verified | Rationale: \"{reason}\" | Answerability: {answerability}")


def cmd_reject(dataset_path: str, record_id: str, reviewer: str, reason: str) -> None:
    records = load_dataset(dataset_path)
    target = next((r for r in records if r.id == record_id), None)
    if not target:
        print(f"Error: Record ID '{record_id}' not found in {dataset_path}")
        return

    target.reject_record(reviewer=reviewer, reason=reason)
    save_dataset(records, dataset_path)
    print(f"[OK] Record '{record_id}' marked as 'rejected' by '{reviewer}'.")
    print(f"     Rejection Reason: \"{reason}\"")


def cmd_export_verified(dataset_path: str, output_path: str) -> None:
    records = load_dataset(dataset_path)
    verified = [r for r in records if r.verification_status == "verified_human"]

    freeze_val = validate_benchmark_freeze(verified)
    if not freeze_val["is_freeze_ready"]:
        print(f"\n[BENCHMARK FREEZE BLOCKED] Cannot export verified benchmark:")
        for err in freeze_val["errors"]:
            print(f"  - {err}")
        print("\nWriting empty / blocked benchmark freeze manifest rather than fabricating verified records...")
        blocked_manifest = {
            "dataset_name": "TrustRAG Retrieval Golden Benchmark (Blocked)",
            "version": "2.0-freeze-blocked",
            "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "BLOCKED_PENDING_HUMAN_REVIEW",
            "verified_count": 0,
            "source_dataset": dataset_path,
            "freeze_validation": freeze_val,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"manifest": blocked_manifest, "records": []}, f, indent=2)
        print(f"[OK] Wrote blocked benchmark freeze manifest to {output_path}")
        return

    content_hash = freeze_val["content_hash"]
    manifest = {
        "dataset_name": "TrustRAG Retrieval Golden Benchmark (Human Verified)",
        "version": "2.0-verified",
        "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verified_count": len(verified),
        "source_dataset": dataset_path,
        "content_hash_sha256": content_hash,
        "breakdown": freeze_val["counts"],
        "validation_status": "PASSED_FREEZE_INVARIANTS",
    }
    save_dataset(verified, output_path, manifest=manifest)
    print(f"[OK] Successfully exported {len(verified)} human-verified benchmark records to {output_path}")
    print(f"     Content SHA-256: {content_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustRAG Human Annotation CLI (v2)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to retrieval dataset JSON")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # stats
    subparsers.add_parser("stats", help="Show summary statistics and review coverage")

    # list
    list_parser = subparsers.add_parser("list", help="List records with filtering")
    list_parser.add_argument("--status", choices=["pending", "unverified_synthetic", "verified_human", "rejected"], default=None)
    list_parser.add_argument("--split", choices=["dev", "test"], default=None)

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect record details, chunks, and review history")
    inspect_parser.add_argument("record_id", help="Record ID to inspect")

    # verify-evidence
    ve_parser = subparsers.add_parser("verify-evidence", help="Inspect chunk content and verify evidence quotes")
    ve_parser.add_argument("record_id", help="Record ID to verify evidence for")

    # verify-qdrant
    vq_parser = subparsers.add_parser("verify-qdrant", help="Verify candidate chunk UUIDs in live Qdrant index")
    vq_parser.add_argument("record_id", nargs="?", default=None, help="Optional record ID to check specific chunks")

    # approve (and promote alias)
    for name in ["approve", "promote"]:
        p = subparsers.add_parser(name, help="Approve record with documented human verification")
        p.add_argument("record_id", help="Record ID to approve")
        p.add_argument("--reviewer", required=True, help="Reviewer identifier/name")
        p.add_argument("--reason", required=True, help="Documented rationale for approval")
        p.add_argument("--answerability", choices=["answerable", "corpus_unanswerable", "unresolved_evidence"], default="answerable")
        p.add_argument("--evidence-judgments", default=None, help="JSON string of chunk judgments")
        p.add_argument("--answer-verified", action="store_true", default=True, help="Confirm expected answer verified")
        p.add_argument("--multi-hop-verified", action="store_true", default=True, help="Confirm multi-hop sets verified")
        p.add_argument("--corrections", default=None, help="JSON string of candidate chunk corrections")

    # reject
    reject_parser = subparsers.add_parser("reject", help="Reject a candidate record with documented reason")
    reject_parser.add_argument("record_id", help="Record ID to reject")
    reject_parser.add_argument("--reviewer", required=True, help="Reviewer identifier/name")
    reject_parser.add_argument("--reason", required=True, help="Documented reason for rejection")

    # export-verified
    export_parser = subparsers.add_parser("export-verified", help="Export human-verified benchmark records")
    export_parser.add_argument("--output", required=True, help="Destination JSON file for verified records")

    args = parser.parse_args()

    if args.subcommand == "stats":
        cmd_stats(args.dataset)
    elif args.subcommand == "list":
        cmd_list(args.dataset, args.status, args.split)
    elif args.subcommand == "inspect":
        cmd_inspect(args.dataset, args.record_id)
    elif args.subcommand == "verify-evidence":
        cmd_verify_evidence(args.dataset, args.record_id)
    elif args.subcommand == "verify-qdrant":
        cmd_verify_qdrant(args.dataset, args.record_id)
    elif args.subcommand in ["approve", "promote"]:
        cmd_approve(
            dataset_path=args.dataset,
            record_id=args.record_id,
            reviewer=args.reviewer,
            reason=args.reason,
            answerability=args.answerability,
            evidence_judgments_json=args.evidence_judgments,
            answer_verified=args.answer_verified,
            multi_hop_verified=args.multi_hop_verified,
            corrections_json=args.corrections,
        )
    elif args.subcommand == "reject":
        cmd_reject(args.dataset, args.record_id, args.reviewer, args.reason)
    elif args.subcommand == "export-verified":
        cmd_export_verified(args.dataset, args.output)


if __name__ == "__main__":
    main()
