"""
TrustRAG Retrieval Dataset Schema & Validator.

Defines the explicit schema for retrieval evaluation records, validates
syntax, detects duplicates, checks split leakage, ensures multi-hop conjunction
integrity, enforces human verification workflows, and verifies chunk UUIDs
against the target Qdrant vector index.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


VALID_QUERY_TYPES = {"answerable_single_hop", "answerable_multi_hop", "unanswerable"}
VALID_UNANSWERABLE_SUBTYPES = {"corpus_unanswerable", "unresolved_evidence"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_VERIFICATION_STATUSES = {"unverified_synthetic", "verified_human", "rejected"}
VALID_SPLITS = {"dev", "test"}

UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)


@dataclass
class AnnotationEvent:
    event_id: str
    reviewer: str
    timestamp: str
    decision: str  # "verified" | "rejected"
    reason: str
    answerability: str = "answerable"
    evidence_judgments: Dict[str, Any] = field(default_factory=dict)
    answer_verified: bool = False
    multi_hop_verified: bool = False
    candidate_corrections: Optional[Dict[str, Any]] = None
    annotation_version: str = "2.0"
    dataset_version: str = "1.0-seed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalEvalRecord:
    id: str
    query: str
    query_type: str
    difficulty: str
    domain: str
    source_files: List[str]
    expected_sections: List[str]
    evidence_quotes: List[str]
    candidate_chunk_ids: List[str]
    relevance_grades: Dict[str, int]
    sufficient_evidence_sets: List[List[str]]
    verification_status: str
    split: str
    unanswerable_subtype: Optional[str] = None
    expected_answer: Optional[str] = None
    reviewer_decision: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    topic_family: Optional[str] = None
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    candidate_corrections: Optional[Dict[str, Any]] = None
    evidence_verification_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievalEvalRecord:
        return cls(
            id=str(data["id"]),
            query=str(data["query"]).strip(),
            query_type=str(data["query_type"]).strip(),
            difficulty=str(data.get("difficulty", "medium")).strip(),
            domain=str(data.get("domain", "enterprise_docs")).strip(),
            source_files=list(data.get("source_files", [])),
            expected_sections=list(data.get("expected_sections", [])),
            evidence_quotes=list(data.get("evidence_quotes", [])),
            candidate_chunk_ids=list(data.get("candidate_chunk_ids", [])),
            relevance_grades={str(k): int(v) for k, v in data.get("relevance_grades", {}).items()},
            sufficient_evidence_sets=[list(s) for s in data.get("sufficient_evidence_sets", [])],
            verification_status=str(data.get("verification_status", "unverified_synthetic")).strip(),
            split=str(data.get("split", "dev")).strip(),
            unanswerable_subtype=data.get("unanswerable_subtype"),
            expected_answer=data.get("expected_answer"),
            reviewer_decision=data.get("reviewer_decision"),
            notes=data.get("notes"),
            topic_family=data.get("topic_family"),
            review_history=list(data.get("review_history", [])),
            candidate_corrections=data.get("candidate_corrections"),
            evidence_verification_details=data.get("evidence_verification_details"),
        )

    def promote_to_verified(
        self,
        reviewer: str,
        reason: str,
        timestamp: Optional[str] = None,
        answerability: str = "answerable",
        evidence_judgments: Optional[Dict[str, Any]] = None,
        answer_verified: bool = True,
        multi_hop_verified: bool = True,
        candidate_corrections: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Promotes an unverified record to verified_human status with reviewer audit record and event history."""
        if not reviewer or not reviewer.strip():
            raise ValueError("Reviewer identity must be specified to promote a record.")
        if not reason or not reason.strip():
            raise ValueError("Review rationale/reason must be provided to promote a record.")
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        evt_id = f"ann_{uuid.uuid4().hex[:12]}"
        self.verification_status = "verified_human"
        self.reviewer_decision = {
            "decision": "verified",
            "reviewer": reviewer.strip(),
            "reason": reason.strip(),
            "timestamp": ts,
            "answerability": answerability,
            "answer_verified": answer_verified,
            "multi_hop_verified": multi_hop_verified,
        }
        event = {
            "event_id": evt_id,
            "reviewer": reviewer.strip(),
            "timestamp": ts,
            "decision": "verified",
            "reason": reason.strip(),
            "answerability": answerability,
            "evidence_judgments": evidence_judgments or {},
            "answer_verified": answer_verified,
            "multi_hop_verified": multi_hop_verified,
            "candidate_corrections": candidate_corrections,
            "annotation_version": "2.0",
        }
        self.review_history.append(event)
        if candidate_corrections:
            self.candidate_corrections = candidate_corrections
        if evidence_judgments:
            self.evidence_verification_details = evidence_judgments

    def reject_record(
        self,
        reviewer: str,
        reason: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Marks a record as rejected with reviewer audit trail and event history."""
        if not reviewer or not reviewer.strip():
            raise ValueError("Reviewer identity must be specified to reject a record.")
        if not reason or not reason.strip():
            raise ValueError("Rejection reason must be provided.")
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        evt_id = f"ann_{uuid.uuid4().hex[:12]}"
        self.verification_status = "rejected"
        self.reviewer_decision = {
            "decision": "rejected",
            "reviewer": reviewer.strip(),
            "reason": reason.strip(),
            "timestamp": ts,
        }
        event = {
            "event_id": evt_id,
            "reviewer": reviewer.strip(),
            "timestamp": ts,
            "decision": "rejected",
            "reason": reason.strip(),
            "annotation_version": "2.0",
        }
        self.review_history.append(event)


def validate_uuid_string(val: str) -> bool:
    """Validates if string conforms to standard UUID format."""
    return bool(UUID_REGEX.match(val.strip()))


def validate_eval_record(record: RetrievalEvalRecord | Dict[str, Any]) -> List[str]:
    """
    Validates a single retrieval record against schema constraints.
    Returns a list of error descriptions (empty list if valid).
    """
    errors: List[str] = []
    r = record if isinstance(record, RetrievalEvalRecord) else RetrievalEvalRecord.from_dict(record)

    # 1. Identifier
    if not r.id or not str(r.id).strip():
        errors.append("Record 'id' must be a non-empty string.")

    # 2. Query
    if not r.query or len(r.query.strip()) < 5:
        errors.append(f"Record {r.id}: 'query' must be at least 5 characters.")

    # 3. Query type
    if r.query_type not in VALID_QUERY_TYPES:
        errors.append(f"Record {r.id}: 'query_type' must be one of {sorted(VALID_QUERY_TYPES)}, got '{r.query_type}'.")

    # 4. Difficulty
    if r.difficulty not in VALID_DIFFICULTIES:
        errors.append(f"Record {r.id}: 'difficulty' must be one of {sorted(VALID_DIFFICULTIES)}, got '{r.difficulty}'.")

    # 5. Verification status
    if r.verification_status not in VALID_VERIFICATION_STATUSES:
        errors.append(f"Record {r.id}: 'verification_status' must be one of {sorted(VALID_VERIFICATION_STATUSES)}, got '{r.verification_status}'.")

    # 6. Human verification audit trail requirement
    if r.verification_status == "verified_human":
        if not r.reviewer_decision:
            errors.append(f"Record {r.id}: cannot be 'verified_human' without a valid 'reviewer_decision'.")
        else:
            rev = r.reviewer_decision.get("reviewer")
            reason = r.reviewer_decision.get("reason")
            if not rev or not str(rev).strip():
                errors.append(f"Record {r.id}: reviewer_decision.reviewer must be specified.")
            if not reason or not str(reason).strip():
                errors.append(f"Record {r.id}: reviewer_decision.reason must be specified.")

    # 7. Split
    if r.split not in VALID_SPLITS:
        errors.append(f"Record {r.id}: 'split' must be one of {sorted(VALID_SPLITS)}, got '{r.split}'.")

    # 8. Unanswerable vs Answerable constraints
    if r.query_type == "unanswerable":
        if len(r.candidate_chunk_ids) > 0:
            errors.append(f"Record {r.id}: unanswerable query must have empty 'candidate_chunk_ids', got {r.candidate_chunk_ids}.")
        if len(r.sufficient_evidence_sets) > 0:
            errors.append(f"Record {r.id}: unanswerable query must have empty 'sufficient_evidence_sets', got {r.sufficient_evidence_sets}.")
        if r.unanswerable_subtype not in VALID_UNANSWERABLE_SUBTYPES:
            errors.append(f"Record {r.id}: unanswerable query must have 'unanswerable_subtype' in {sorted(VALID_UNANSWERABLE_SUBTYPES)}, got '{r.unanswerable_subtype}'.")
    else:
        # Answerable queries must have candidate chunks
        if len(r.candidate_chunk_ids) == 0:
            errors.append(f"Record {r.id}: answerable query must specify at least one relevant chunk ID in 'candidate_chunk_ids'.")
        if len(r.sufficient_evidence_sets) == 0:
            errors.append(f"Record {r.id}: answerable query must specify at least one sufficient evidence set.")
        if len(r.evidence_quotes) == 0:
            errors.append(f"Record {r.id}: answerable query should provide at least one evidence quote.")

    # 9. UUID validity for candidate chunks
    for cid in r.candidate_chunk_ids:
        if not validate_uuid_string(cid):
            errors.append(f"Record {r.id}: chunk ID '{cid}' is not a valid UUID format.")

    # 10. Conjunction integrity: every chunk in sufficient_evidence_sets must be in candidate_chunk_ids
    cand_set = set(r.candidate_chunk_ids)
    for idx, s_set in enumerate(r.sufficient_evidence_sets):
        if not s_set:
            errors.append(f"Record {r.id}: sufficient_evidence_set[{idx}] is empty.")
        for cid in s_set:
            if cid not in cand_set:
                errors.append(f"Record {r.id}: chunk ID '{cid}' in sufficient_evidence_set[{idx}] missing from candidate_chunk_ids.")

    # 11. Multi-hop rule: must require multiple chunks in at least one set
    if r.query_type == "answerable_multi_hop":
        all_single = all(len(s) <= 1 for s in r.sufficient_evidence_sets)
        if all_single:
            errors.append(f"Record {r.id}: multi-hop query must have at least one sufficient set with >1 chunk.")

    # 12. Relevance grades: all candidate chunks should have valid grade (0 to 3)
    for cid in r.candidate_chunk_ids:
        grade = r.relevance_grades.get(cid, 1)
        if grade < 0 or grade > 3:
            errors.append(f"Record {r.id}: grade for chunk '{cid}' must be between 0 and 3, got {grade}.")

    return errors


def validate_dataset_records(records: List[RetrievalEvalRecord | Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates a list of retrieval records, checks for duplicates, checks split leakage,
    and produces a comprehensive breakdown.
    """
    all_errors: Dict[str, List[str]] = {}
    seen_ids: Set[str] = set()
    seen_queries: Dict[str, str] = {}
    duplicates: List[Dict[str, str]] = []
    split_leakage: List[Dict[str, Any]] = []

    dev_queries: Dict[str, str] = {}
    test_queries: Dict[str, str] = {}
    dev_single_chunk_sets: Dict[str, str] = {}
    test_single_chunk_sets: Dict[str, str] = {}

    parsed_records: List[RetrievalEvalRecord] = []
    for raw in records:
        rec = raw if isinstance(raw, RetrievalEvalRecord) else RetrievalEvalRecord.from_dict(raw)
        parsed_records.append(rec)

        # Check unique ID
        if rec.id in seen_ids:
            all_errors.setdefault(rec.id, []).append(f"Duplicate record ID: '{rec.id}'.")
        else:
            seen_ids.add(rec.id)

        # Check duplicate query text across entire dataset
        norm_q = rec.query.strip().lower()
        if norm_q in seen_queries:
            duplicates.append({"original_id": seen_queries[norm_q], "duplicate_id": rec.id, "query": rec.query})
        else:
            seen_queries[norm_q] = rec.id

        # Populate split trackers
        if rec.split == "dev":
            dev_queries[norm_q] = rec.id
            if rec.query_type == "answerable_single_hop" and len(rec.candidate_chunk_ids) == 1:
                dev_single_chunk_sets[rec.candidate_chunk_ids[0]] = rec.id
        elif rec.split == "test":
            test_queries[norm_q] = rec.id
            if rec.query_type == "answerable_single_hop" and len(rec.candidate_chunk_ids) == 1:
                test_single_chunk_sets[rec.candidate_chunk_ids[0]] = rec.id

        # Validate single record constraints
        errs = validate_eval_record(rec)
        if errs:
            all_errors[rec.id] = errs

    # Check split leakage:
    # 1. Verbatim query across dev and test
    overlap_queries = set(dev_queries.keys()) & set(test_queries.keys())
    for oq in overlap_queries:
        split_leakage.append({
            "type": "verbatim_query_leakage",
            "query": oq,
            "dev_id": dev_queries[oq],
            "test_id": test_queries[oq],
        })

    breakdown = {
        "total": len(parsed_records),
        "by_type": {
            "answerable_single_hop": sum(1 for r in parsed_records if r.query_type == "answerable_single_hop"),
            "answerable_multi_hop": sum(1 for r in parsed_records if r.query_type == "answerable_multi_hop"),
            "unanswerable": sum(1 for r in parsed_records if r.query_type == "unanswerable"),
        },
        "by_unanswerable_subtype": {
            "corpus_unanswerable": sum(1 for r in parsed_records if r.unanswerable_subtype == "corpus_unanswerable"),
            "unresolved_evidence": sum(1 for r in parsed_records if r.unanswerable_subtype == "unresolved_evidence"),
        },
        "by_verification": {
            "unverified_synthetic": sum(1 for r in parsed_records if r.verification_status == "unverified_synthetic"),
            "verified_human": sum(1 for r in parsed_records if r.verification_status == "verified_human"),
            "rejected": sum(1 for r in parsed_records if r.verification_status == "rejected"),
        },
        "by_split": {
            "dev": sum(1 for r in parsed_records if r.split == "dev"),
            "test": sum(1 for r in parsed_records if r.split == "test"),
        },
    }

    is_valid = (len(all_errors) == 0) and (len(duplicates) == 0) and (len(split_leakage) == 0)

    return {
        "is_valid": is_valid,
        "total_records": len(parsed_records),
        "breakdown": breakdown,
        "errors": all_errors,
        "duplicates": duplicates,
        "split_leakage": split_leakage,
    }


def validate_records_against_qdrant(
    records: List[RetrievalEvalRecord | Dict[str, Any]],
    qdrant_client: Any,
    collection_name: str
) -> Dict[str, Any]:
    """
    Validates that every candidate_chunk_id in the dataset actually exists in
    the target Qdrant collection.
    """
    all_chunk_ids: Set[str] = set()
    for raw in records:
        rec = raw if isinstance(raw, RetrievalEvalRecord) else RetrievalEvalRecord.from_dict(raw)
        for cid in rec.candidate_chunk_ids:
            all_chunk_ids.add(cid)

    if not all_chunk_ids:
        return {"total_checked": 0, "found": 0, "missing": [], "all_verified": True}

    chunk_list = list(all_chunk_ids)
    pts = qdrant_client.retrieve(collection_name=collection_name, ids=chunk_list, with_payload=False)
    found_ids = {str(pt.id) for pt in pts}
    missing = [cid for cid in chunk_list if cid not in found_ids]

    return {
        "total_checked": len(all_chunk_ids),
        "found": len(found_ids),
        "missing": missing,
        "all_verified": len(missing) == 0,
    }


def load_dataset(file_path: str) -> List[RetrievalEvalRecord]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records_data = data.get("records", data) if isinstance(data, dict) else data
    return [RetrievalEvalRecord.from_dict(d) for d in records_data]


def save_dataset(records: List[RetrievalEvalRecord | Dict[str, Any]], file_path: str, manifest: Optional[Dict[str, Any]] = None) -> None:
    dict_records = [r.to_dict() if isinstance(r, RetrievalEvalRecord) else r for r in records]
    payload = {
        "version": "1.0",
        "manifest": manifest or {},
        "records": dict_records,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_records_content_hash(records: List[RetrievalEvalRecord | Dict[str, Any]]) -> str:
    """Computes a deterministic SHA-256 content hash of canonical records representation."""
    serialized = []
    for r in records:
        d = r.to_dict() if isinstance(r, RetrievalEvalRecord) else dict(r)
        canonical = {k: d[k] for k in sorted(d.keys())}
        serialized.append(canonical)
    payload_str = json.dumps(serialized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def resolve_chunk_evidence(chunk_id: str, corpus_dir: str = "processed_data/true") -> Optional[Dict[str, Any]]:
    """
    Resolves a candidate chunk UUID against local true corpus JSON files.
    Returns dict with {uuid, filename, chunk_index, section, text, content_hash} or None.
    """
    import glob
    from app.config import settings
    ns = uuid.UUID(settings.PROJECT_UUID_NAMESPACE)

    for fpath in glob.glob(os.path.join(corpus_dir, "*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            fn = data.get("filename", os.path.basename(fpath).replace(".json", ""))
            for c in data.get("chunks", []):
                c_idx = c["chunk_index"]
                c_text = c["text"]
                c_hash = hashlib.sha256(c_text.strip().encode("utf-8")).hexdigest()[:16]
                key = f"{fn}:{c_idx}:{c_hash}"
                c_uuid = str(uuid.uuid5(ns, key))
                if c_uuid == chunk_id:
                    return {
                        "uuid": c_uuid,
                        "filename": fn,
                        "chunk_index": c_idx,
                        "section": c.get("section", "General"),
                        "text": c_text,
                        "content_hash": c_hash,
                    }
        except Exception:
            continue
    return None


def validate_benchmark_freeze(records: List[RetrievalEvalRecord | Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enforces strict benchmark freeze invariants.
    Fails if:
    1. Zero approved records exist (empty/blocked export).
    2. Any unverified or rejected record is present in proposed export.
    3. Required review fields (reviewer, reason, timestamp) are missing.
    4. Evidence references fail schema/UUID validation.
    5. Split integrity is violated (verbatim query leakage across dev/test).
    """
    errors: List[str] = []
    warnings: List[str] = []

    parsed: List[RetrievalEvalRecord] = [
        r if isinstance(r, RetrievalEvalRecord) else RetrievalEvalRecord.from_dict(r)
        for r in records
    ]

    if not parsed:
        errors.append("Benchmark freeze failed: 0 approved records provided. Cannot freeze an empty benchmark.")
        return {
            "is_freeze_ready": False,
            "total_approved": 0,
            "content_hash": "",
            "errors": errors,
            "warnings": warnings,
            "counts": {},
        }

    dev_queries: Set[str] = set()
    test_queries: Set[str] = set()

    for r in parsed:
        # Must be strictly verified_human
        if r.verification_status != "verified_human":
            errors.append(f"Record {r.id}: has verification_status='{r.verification_status}'. Only 'verified_human' permitted in benchmark freeze.")

        # Must have valid reviewer_decision
        if not r.reviewer_decision:
            errors.append(f"Record {r.id}: missing required 'reviewer_decision' audit record.")
        else:
            rev = r.reviewer_decision.get("reviewer")
            reason = r.reviewer_decision.get("reason")
            ts = r.reviewer_decision.get("timestamp")
            if not rev or not str(rev).strip():
                errors.append(f"Record {r.id}: reviewer_decision must specify non-empty 'reviewer'.")
            if not reason or not str(reason).strip():
                errors.append(f"Record {r.id}: reviewer_decision must specify non-empty 'reason'.")
            if not ts or not str(ts).strip():
                errors.append(f"Record {r.id}: reviewer_decision must specify valid 'timestamp'.")

        # Schema checks
        s_errs = validate_eval_record(r)
        if s_errs:
            errors.extend(s_errs)

        # Track query leakage
        norm_q = " ".join(r.query.strip().lower().split())
        if r.split == "dev":
            dev_queries.add(norm_q)
        elif r.split == "test":
            test_queries.add(norm_q)

    # Verbatim query leakage check
    overlap = dev_queries & test_queries
    if overlap:
        errors.append(f"Benchmark freeze failed: {len(overlap)} verbatim queries appear in both dev and test.")

    content_hash = compute_records_content_hash(parsed) if not errors else ""

    counts = {
        "total": len(parsed),
        "by_split": {
            "dev": sum(1 for r in parsed if r.split == "dev"),
            "test": sum(1 for r in parsed if r.split == "test"),
        },
        "by_type": {
            "answerable_single_hop": sum(1 for r in parsed if r.query_type == "answerable_single_hop"),
            "answerable_multi_hop": sum(1 for r in parsed if r.query_type == "answerable_multi_hop"),
            "unanswerable": sum(1 for r in parsed if r.query_type == "unanswerable"),
        },
    }

    return {
        "is_freeze_ready": len(errors) == 0,
        "total_approved": len(parsed),
        "content_hash": content_hash,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }

