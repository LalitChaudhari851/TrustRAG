# Dataset Integrity & Annotation Eligibility Audit

**Dataset File:** `evals/data/retrieval_golden_seed_v1.json`  
**Dataset SHA256:** `0b63bf1b40684ff7ef18e3fa63b7edb31d628f339bc71d2dea7e023fbe798451`  
**Record Range:** `seed_001` through `seed_050` (Strict sequential 50 questions)  
**Total Selected:** 50 records  
**Audit Timestamp:** 2026-10-26T23:52:24.773392

---

## 1. ID Uniqueness & Sequential Ordering
- IDs present: `seed_001` to `seed_050`
- Unique IDs: 50 / 50 (100% Unique)
- Missing / Skipped Records: 0

---

## 2. Cohort Separation & Verification Eligibility

| Cohort Name | ID Range | Count | Declared Verification Status | Metric Eligibility Status |
| :--- | :---: | :---: | :---: | :--- |
| **Batch 1 (Disputed Mapping)** | `seed_001`–`seed_005` | 5 | `verified_human` | **Disputed:** Approval history points to Kubernetes architecture / legacy mapping. Pending reconciliation before trusted acceptance. |
| **Batch 2 (Verified Corpus Edits)** | `seed_006`–`seed_010` | 5 | `unverified_synthetic` | **Provisional:** Source text & chunk IDs audited against live corpus (`monitor_job.docx`, `cronjobs.docx`, `pods_autoscale.html`), pending formal human sign-off. |
| **Batch 3 (Synthetic Remainder)** | `seed_011`–`seed_050` | 40 | `unverified_synthetic` | **Unverified:** Fully synthetic labels generated during pipeline bootstrap. Contains single-hop, multi-hop, and unanswerable questions. |

---

## 3. Query Type Distribution (All 50 Records)
- **answerable_single_hop**: 30 questions (60.0%)
- **answerable_multi_hop**: 10 questions (20.0%)
- **unanswerable**: 10 questions (20.0%)

> [!CAUTION]
> Because 45 of 50 records are `unverified_synthetic` and 5 have disputed mappings, formal IR retrieval benchmarks cannot claim production sign-off based solely on automated grades. All metrics are presented with strict cohort stratification.
