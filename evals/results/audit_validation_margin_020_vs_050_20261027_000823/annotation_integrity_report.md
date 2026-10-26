# Annotation Integrity & Provenance Audit Report

**Audit Date:** 2026-10-27 00:08:23  
**Source Dataset:** `evals/data/retrieval_golden_seed_v1.json`  
**Verified Manifest:** `evals/data/retrieval_golden_verified.json`  
**Dataset SHA256:** `0b63bf1b40684ff7ef18e3fa63b7edb31d628f339bc71d2dea7e023fbe798451`  

---

## 1. Provenance & Verification Audit

### 1.1 The `seed_001`–`seed_005` Mapping Discrepancy
- **Background:** `seed_001` through `seed_005` carry `"verification_status": "verified_human"` in `retrieval_golden_seed_v1.json`. However, previous evaluation logs referenced Kubernetes Architecture and CronJob questions for these IDs, while the current dataset contains Redis work queue questions (`parallel_work_queue.txt`).
- **Audit Findings:**
  1. The questions currently in `retrieval_golden_seed_v1.json` (`seed_001`–`seed_005`) are valid, coherent, and cite existing chunks (`b9f7708c-ec05-517b-9e33-b33cd86aa109`, etc.).
  2. The verified manifest (`retrieval_golden_verified.json`) correctly lists `status: BLOCKED_PENDING_HUMAN_REVIEW` with `verified_count: 0`.
  3. **Strict Guideline Adherence:** We DO NOT treat `seed_001`–`seed_005` as verified ground truth until a human reviewer completes an explicit re-audit and logs a re-approval event.

### 1.2 Batch 2 (`seed_006`–`seed_010`) Status
- These 5 questions were rewritten and refined to strictly reflect corpus documents (`monitor_job.docx`, `cronjobs.docx`, `pods_autoscale.html`).
- All 5 chunks exist in Qdrant and contain verbatim supporting quotes.
- Verification status remains `unverified_synthetic` until human review.

### 1.3 Batch 3 (`seed_011`–`seed_050`) Synthetic Distribution
- 40 records generated synthetically during pipeline setup.
- Breakdown:
  - 18 Answerable Single-Hop
  - 10 Answerable Multi-Hop
  - 12 Unanswerable Out-of-Domain / Corpus-Absent queries
- Status: Fully synthetic. No claims of human verification.

---

## 2. Evidence UUID Alignment Against Active Qdrant Snapshot

All candidate chunk UUIDs referenced in `retrieval_golden_seed_v1.json` for answerable questions were verified against the live Qdrant `enterprise_rag_v2` collection:
- `b9f7708c-ec05-517b-9e33-b33cd86aa109` (Redis Pod/Service) $	o$ Present in Qdrant (Chunk 0)
- `83358e9c-431c-51b4-97e8-baacb2d11344` (Parallelism: 2) $	o$ Present in Qdrant (Chunk 5)
- `737920f2-0936-53da-82e8-e2a63c86ba07` (Job Restart Policies) $	o$ Present in Qdrant (Chunk 6)
- `4e38baae-c5a4-5da9-86ce-b1ef65d8a9e7` (`backoffLimit`) $	o$ Present in Qdrant (Chunk 7)
- `47b71360-15cb-54d9-95e2-0fb34c6778f6` (`activeDeadlineSeconds`) $	o$ Present in Qdrant (Chunk 25)
- `8f7910c9-d6a1-5234-950f-ed26139ea9a3` (`ttlSecondsAfterFinished: 3600`) $	o$ Present in Qdrant (Chunk 52)
- `ac2984be-e47e-5b11-937a-46344e9fcfed` (HPA vs VPA) $	o$ Present in Qdrant (Chunk 27)

Zero orphaned UUIDs were found in the golden dataset.

---

## 3. Preservation Invariants
- No golden dataset files were modified.
- No reviewer signatures or timestamps were fabricated.
- Previous benchmark reports and results are preserved intact.
