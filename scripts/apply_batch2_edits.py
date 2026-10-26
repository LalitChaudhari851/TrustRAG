import json
import os
import copy
from evals.retrieval_dataset_schema import validate_eval_record, validate_dataset_records

path = "evals/data/retrieval_golden_seed_v1.json"
with open(path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

records = dataset["records"]
assert len(records) == 50, f"Expected 50 records, got {len(records)}"

# Safeguard check: seed_001 - seed_005 must be verified_human
for i in range(5):
    assert records[i]["id"] == f"seed_00{i+1}"
    assert records[i]["verification_status"] == "verified_human"
    assert records[i]["reviewer_decision"] is not None

# The exact 5 updates
batch2_updates = {
    "seed_006": {
        "id": "seed_006",
        "query": "What does the backoffLimit field specify in a Kubernetes Job configuration?",
        "query_type": "answerable_single_hop",
        "difficulty": "medium",
        "domain": "kubernetes_workloads",
        "source_files": [
            "monitor_job.docx"
        ],
        "expected_sections": [
            "Creating a Job Configuration File"
        ],
        "evidence_quotes": [
            "Has a backoffLimit of 4, meaning it will retry up to 4 times if it fails"
        ],
        "candidate_chunk_ids": [
            "50ebc9c7-ac56-52ec-8a80-225d798fff1d"
        ],
        "relevance_grades": {
            "50ebc9c7-ac56-52ec-8a80-225d798fff1d": 3
        },
        "sufficient_evidence_sets": [
            [
                "50ebc9c7-ac56-52ec-8a80-225d798fff1d"
            ]
        ],
        "verification_status": "unverified_synthetic",
        "split": "dev",
        "unanswerable_subtype": None,
        "expected_answer": "It specifies how many times a Job will retry if it fails.",
        "reviewer_decision": None,
        "notes": "Rewritten single-hop query grounded strictly in monitor_job.docx Chunk 7; pending human review.",
        "topic_family": None,
        "review_history": [],
        "candidate_corrections": {
            "original_query": "What is the default backoffLimit for a Kubernetes Job?",
            "original_candidate_chunk_ids": [
                "81a4f786-ef7a-52e5-9af0-339d7dd76240"
            ],
            "replacement_candidate_chunks": [
                "50ebc9c7-ac56-52ec-8a80-225d798fff1d"
            ],
            "reason": "Default backoffLimit (6) was missing from corpus. Rewritten to evaluate what backoffLimit specifies based on monitor_job.docx Chunk 7."
        },
        "evidence_verification_details": None
    },
    "seed_007": {
        "id": "seed_007",
        "query": "What does activeDeadlineSeconds do when applied to a Job?",
        "query_type": "answerable_single_hop",
        "difficulty": "hard",
        "domain": "kubernetes_workloads",
        "source_files": [
            "cronjobs.docx"
        ],
        "expected_sections": [
            "4. Jobs Taking Too Long"
        ],
        "evidence_quotes": [
            "# Set active deadline\nspec:\nactiveDeadlineSeconds: 3600  # Kill job after 1 hour"
        ],
        "candidate_chunk_ids": [
            "84a9f778-e252-530b-9d2e-9aade0158d16"
        ],
        "relevance_grades": {
            "84a9f778-e252-530b-9d2e-9aade0158d16": 3
        },
        "sufficient_evidence_sets": [
            [
                "84a9f778-e252-530b-9d2e-9aade0158d16"
            ]
        ],
        "verification_status": "unverified_synthetic",
        "split": "dev",
        "unanswerable_subtype": None,
        "expected_answer": "It sets an active execution deadline that terminates (kills) the Job after the specified duration (e.g., 3600 seconds) if it runs longer than expected.",
        "reviewer_decision": None,
        "notes": "Evidence corrected to cronjobs.docx Chunk 25; pending human review.",
        "topic_family": None,
        "review_history": [],
        "candidate_corrections": {
            "original_candidate_chunk_ids": [
                "dd8b3b09-eaf6-59dd-b6d3-1844d69b8901"
            ],
            "replacement_candidate_chunks": [
                "84a9f778-e252-530b-9d2e-9aade0158d16"
            ],
            "reason": "Replaced irrelevant Chunk 7 (NonIndexed) with Chunk 25 (Jobs Taking Too Long) which directly defines activeDeadlineSeconds."
        },
        "evidence_verification_details": None
    },
    "seed_008": {
        "id": "seed_008",
        "query": "How can you configure a Kubernetes CronJob to automatically clean up completed jobs after 1 hour to minimize costs?",
        "query_type": "answerable_single_hop",
        "difficulty": "medium",
        "domain": "kubernetes_workloads",
        "source_files": [
            "cronjobs.docx"
        ],
        "expected_sections": [
            "2. Schedule Optimization for Cost Savings"
        ],
        "evidence_quotes": [
            "# Quick cleanup to minimize costs\nttlSecondsAfterFinished: 3600  # Clean up after 1 hour"
        ],
        "candidate_chunk_ids": [
            "8f7910c9-d6a1-5234-950f-ed26139ea9a3"
        ],
        "relevance_grades": {
            "8f7910c9-d6a1-5234-950f-ed26139ea9a3": 3
        },
        "sufficient_evidence_sets": [
            [
                "8f7910c9-d6a1-5234-950f-ed26139ea9a3"
            ]
        ],
        "verification_status": "unverified_synthetic",
        "split": "dev",
        "unanswerable_subtype": None,
        "expected_answer": "Set ttlSecondsAfterFinished: 3600 in the CronJob job template specification for quick cleanup 1 hour after completion.",
        "reviewer_decision": None,
        "notes": "Rewritten to ground strictly in cronjobs.docx Chunk 52 cleanup configuration; pending human review.",
        "topic_family": None,
        "review_history": [],
        "candidate_corrections": {
            "original_query": "How does ttlSecondsAfterFinished clean up completed Kubernetes Jobs?",
            "original_candidate_chunk_ids": [
                "28cb15bb-064d-5f01-a0fc-0b0c718ed336"
            ],
            "replacement_candidate_chunks": [
                "8f7910c9-d6a1-5234-950f-ed26139ea9a3"
            ],
            "reason": "Original question asked for internal TTL controller mechanics absent from text. Rewritten to evaluate documented 1-hour cleanup configuration in Chunk 52."
        },
        "evidence_verification_details": None
    },
    "seed_009": {
        "id": "seed_009",
        "query": "What is the core difference between HPA and VPA in Kubernetes?",
        "query_type": "answerable_single_hop",
        "difficulty": "easy",
        "domain": "kubernetes_workloads",
        "source_files": [
            "pods_autoscale.html"
        ],
        "expected_sections": [
            "HPA vs. VPA: Which One to Use?"
        ],
        "evidence_quotes": [
            "- HPA is ideal for stateless applications that can easily scale horizontally by adding more pods. It’s useful when you need to handle variable workloads and ensure high availability.\n- VPA is suitable for applications where scaling horizontally is not feasible or when you need to optimize resource usage for individual pods. It’s beneficial for stateful applications or those with varying resource needs."
        ],
        "candidate_chunk_ids": [
            "ac2984be-e47e-5b11-937a-46344e9fcfed",
            "376b124c-3b4c-53e2-81d6-28ae65918ad9",
            "c0f1220e-829c-50f0-bf0d-2e48be167355"
        ],
        "relevance_grades": {
            "ac2984be-e47e-5b11-937a-46344e9fcfed": 3,
            "376b124c-3b4c-53e2-81d6-28ae65918ad9": 2,
            "c0f1220e-829c-50f0-bf0d-2e48be167355": 1
        },
        "sufficient_evidence_sets": [
            [
                "ac2984be-e47e-5b11-937a-46344e9fcfed"
            ]
        ],
        "verification_status": "unverified_synthetic",
        "split": "dev",
        "unanswerable_subtype": None,
        "expected_answer": "HPA scales horizontally by adding more pods. VPA adjusts CPU and memory requests and limits for individual pods.",
        "reviewer_decision": None,
        "notes": "Evidence corrected to pods_autoscale.html Chunk 27 (Grade 3), Chunk 13 (Grade 2 for CPU/mem adjustment details), and Chunk 1 (Grade 1); pending human review.",
        "topic_family": None,
        "review_history": [],
        "candidate_corrections": {
            "added_candidate_chunks": [
                "ac2984be-e47e-5b11-937a-46344e9fcfed",
                "376b124c-3b4c-53e2-81d6-28ae65918ad9"
            ],
            "regraded_candidate_chunks": {
                "ac2984be-e47e-5b11-937a-46344e9fcfed": 3,
                "376b124c-3b4c-53e2-81d6-28ae65918ad9": 2,
                "c0f1220e-829c-50f0-bf0d-2e48be167355": 1
            },
            "reason": "Chunk 27 contrasts scaling by adding pods vs individual pods; Chunk 13 defines VPA adjusting CPU and memory requests and limits."
        },
        "evidence_verification_details": None
    },
    "seed_010": {
        "id": "seed_010",
        "query": "How do you install the Metrics Server in Kubernetes using the official components manifest?",
        "query_type": "answerable_single_hop",
        "difficulty": "medium",
        "domain": "kubernetes_workloads",
        "source_files": [
            "pods_autoscale.html"
        ],
        "expected_sections": [
            "Metric Server Setup"
        ],
        "evidence_quotes": [
            "$ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
        ],
        "candidate_chunk_ids": [
            "6abf9610-2c0e-589b-b750-9721ae48209d"
        ],
        "relevance_grades": {
            "6abf9610-2c0e-589b-b750-9721ae48209d": 3
        },
        "sufficient_evidence_sets": [
            [
                "6abf9610-2c0e-589b-b750-9721ae48209d"
            ]
        ],
        "verification_status": "unverified_synthetic",
        "split": "dev",
        "unanswerable_subtype": None,
        "expected_answer": "Apply the components manifest directly from the official GitHub release URL: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml.",
        "reviewer_decision": None,
        "notes": "Rewritten to evaluate official Metrics Server manifest installation in pods_autoscale.html Chunk 2; pending human review.",
        "topic_family": None,
        "review_history": [],
        "candidate_corrections": {
            "original_query": "Which insecure TLS argument is commonly added to Metrics Server in test clusters?",
            "reason": "Original query asked for --kubelet-insecure-tls which is absent from corpus. Rewritten to evaluate the documented kubectl apply installation command in Chunk 2."
        },
        "evidence_verification_details": None
    }
}

# Apply updates to indices 5 to 9 (seed_006 to seed_010)
for idx, target_id in enumerate(["seed_006", "seed_007", "seed_008", "seed_009", "seed_010"], start=5):
    assert records[idx]["id"] == target_id, f"Index mismatch: expected {target_id}, got {records[idx]['id']}"
    records[idx] = batch2_updates[target_id]

# Validate all records before saving
for r in records:
    errs = validate_eval_record(r)
    if errs:
        raise ValueError(f"Record {r.get('id')} validation failed: {errs}")

val_res = validate_dataset_records(records)
if not val_res["is_valid"]:
    raise ValueError(f"Dataset-level validation failed: {val_res['errors']}")

# Write updated dataset
with open(path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Successfully applied Batch 2 refined edits to retrieval_golden_seed_v1.json")
