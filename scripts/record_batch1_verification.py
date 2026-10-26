import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.retrieval_dataset_schema import (
    load_dataset,
    save_dataset,
    validate_dataset_records,
    validate_eval_record,
    resolve_chunk_evidence,
)

DATASET_PATH = "evals/data/retrieval_golden_seed_v1.json"
REVIEWER_NAME = "Lalit Chaudhari"

def main():
    records = load_dataset(DATASET_PATH)
    target_ids = {"seed_001", "seed_002", "seed_003", "seed_004", "seed_005"}

    # Track pre-state for validation
    pre_statuses = {r.id: r.verification_status for r in records}

    # 1. seed_001
    r1 = next(r for r in records if r.id == "seed_001")
    r1.relevance_grades["b9f7708c-ec05-517b-9e33-b33cd86aa109"] = 3
    r1.promote_to_verified(
        reviewer=REVIEWER_NAME,
        reason="Verified evidence chunk in parallel_work_queue.txt [STEP 1: STARTING REDIS]; directly answers how to start Redis using kubectl apply.",
        answerability="answerable",
        evidence_judgments={"b9f7708c-ec05-517b-9e33-b33cd86aa109": {"grade": 3, "status": "verified_exact_match"}},
        answer_verified=True,
        multi_hop_verified=False,
    )

    # 2. seed_002
    r2 = next(r for r in records if r.id == "seed_002")
    r2.relevance_grades["737920f2-0936-53da-82e8-e2a63c86ba07"] = 3
    r2.promote_to_verified(
        reviewer=REVIEWER_NAME,
        reason="Verified evidence chunk in parallel_work_queue.txt [STEP 2: FILLING THE QUEUE WITH TASKS]; directly answers CLI commands for redis-cli and rpush job2.",
        answerability="answerable",
        evidence_judgments={"737920f2-0936-53da-82e8-e2a63c86ba07": {"grade": 3, "status": "verified_exact_match"}},
        answer_verified=True,
        multi_hop_verified=False,
    )

    # 3. seed_003
    r3 = next(r for r in records if r.id == "seed_003")
    r3.relevance_grades["38bfbdef-fa3f-542a-b5c0-6f739d9004cc"] = 3
    r3.promote_to_verified(
        reviewer=REVIEWER_NAME,
        reason="Verified evidence chunk in parallel_work_queue.txt [STEP 4: DEFINING A JOB]; manifest explicitly specifies spec.parallelism: 2.",
        answerability="answerable",
        evidence_judgments={"38bfbdef-fa3f-542a-b5c0-6f739d9004cc": {"grade": 3, "status": "verified_exact_match"}},
        answer_verified=True,
        multi_hop_verified=False,
    )

    # 4. seed_004
    r4 = next(r for r in records if r.id == "seed_004")
    r4.relevance_grades["134fb45c-5ffa-5fd2-b5b8-02c405f097d9"] = 3
    r4.evidence_quotes = [
        "Jobs have strict requirements for Pod restart policies:\n"
        "OnFailure: Kubelet restarts the container within the same Pod if it crashes.\n"
        "Never: Entire Pod is marked as failed, Job controller creates a new Pod.\n"
        "Always: Not allowed (would prevent Pod from ever completing)."
    ]
    r4.promote_to_verified(
        reviewer=REVIEWER_NAME,
        reason="Verified evidence chunk in cronjobs.docx [1. Restart Policy Requirements]; explicitly specifies OnFailure, Never, and that Always is not allowed.",
        answerability="answerable",
        evidence_judgments={"134fb45c-5ffa-5fd2-b5b8-02c405f097d9": {"grade": 3, "status": "verified_exact_match"}},
        answer_verified=True,
        multi_hop_verified=False,
        candidate_corrections={"corrected_quote": "Replaced paraphrased synthetic quote with exact corpus wording from cronjobs.docx"},
    )

    # 5. seed_005
    r5 = next(r for r in records if r.id == "seed_005")
    r5.relevance_grades["1c470929-b8ba-52dc-afd7-ff41adcd2672"] = 3
    r5.evidence_quotes = [
        "Belongs to the batch/v1 API group (stable)."
    ]
    r5.promote_to_verified(
        reviewer=REVIEWER_NAME,
        reason="Verified evidence chunk in cronjobs.docx [2. API Group and Immutability]; explicitly confirms Job belongs to batch/v1 API group.",
        answerability="answerable",
        evidence_judgments={"1c470929-b8ba-52dc-afd7-ff41adcd2672": {"grade": 3, "status": "verified_exact_match"}},
        answer_verified=True,
        multi_hop_verified=False,
        candidate_corrections={"corrected_quote": "Replaced fabricated synthetic YAML snippet with verbatim sentence 'Belongs to the batch/v1 API group (stable).'"},
    )

    # Validation checks
    val = validate_dataset_records(records)
    if not val["is_valid"]:
        print(f"[ERROR] Dataset validation failed: {val['errors']}")
        sys.exit(1)

    # Verify only target_ids changed status
    changed_ids = set()
    for r in records:
        if pre_statuses[r.id] != r.verification_status:
            changed_ids.add(r.id)

    if changed_ids != target_ids:
        print(f"[ERROR] Expected changed IDs {target_ids}, but got {changed_ids}")
        sys.exit(1)

    # Save dataset
    save_dataset(records, DATASET_PATH)
    print(f"[SUCCESS] Successfully recorded human verification for: {sorted(target_ids)}")
    print(f"Total verified in dataset: {sum(1 for r in records if r.verification_status == 'verified_human')} / {len(records)}")

if __name__ == '__main__':
    main()
