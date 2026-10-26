# TrustRAG 50-Question Retrieval Audit Summary

**Audit Date:** 2026-10-27 00:08:23  
**Target Evaluation Run:** `evals/results/validation_margin_020_vs_050_20261026_234441_9068/`  
**Audit Artifacts Directory:** `evals/results/audit_validation_margin_020_vs_050_20261027_000823/`  
**Dataset Reference:** `evals/data/retrieval_golden_seed_v1.json` (SHA256: `0b63bf1b40684ff7...`)  

---

## 1. Executive Audit Findings & Corrected A/B Performance

The automated audit reassessed all 50 question pairs against source documents, chunk UUIDs, full LLM traces, and scoring rules. 

### Corrected Headline Numbers

| Metric | Original Variant A (0.20) | Original Variant B (0.50) | Audited Variant A (0.20) | Audited Variant B (0.50) | Corrected Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pass Rate (All 50)** | 52.0% (26/50) | 54.0% (27/50) | **58.0%** (29/50) | **64.0%** (32/50) | **+6.0% (+3 fixes)** |
| **Confirmed Fixes in B** | — | 2 (`seed_008`, `009`) | — | **3** (`seed_008`, `009`, `040`) | **+3 questions** |
| **Confirmed Regressions** | — | 1 (`seed_030`) | — | **0** (False regression debunked) | **0 regressions** |
| **False Negatives Corrected**| — | — | +3 questions | +3 questions | — |
| **False Abstentions Corrected**| — | — | +1 (`seed_021`) | +2 (`seed_021`, `seed_040`)| — |

---

## 2. Deep-Dive Audit of Target Categories

### 2.1 Incorrect Answers (seed_014, seed_026)
- **`seed_014` (Worker node container agent):**
  - Query: *"What component on worker nodes ensures containers are running in a Pod?"*
  - Expected: *"The kubelet agent ensures containers described in PodSpecs are running and healthy."*
  - App Answer (both A & B): *"Based on the provided technical context, the **kubelet** is the component that acts as the node agent responsible for managing pod lifecycle on its host."*
  - **Audit Verdict:** **FALSE NEGATIVE**. The model answered correctly with the exact entity (`kubelet`) and function. Graded INCORRECT due to uncalibrated token-overlap on expected phrasing. Corrected to **CORRECT** in both variants.
- **`seed_026` (Kube-scheduler responsibility):**
  - Expected: *"Assigning newly created pods with no assigned node to an appropriate worker node."*
  - App Answer: *"evaluates workload requirements and attempts to place them on a matching resource."*
  - **Audit Verdict:** Paraphrased summary from slide corpus. While conceptually accurate, it omits the explicit pod-to-node assignment phrasing. Maintained as **INCORRECT / PARTIAL** for human review.

### 2.2 Partial Answers & Suspected False Regressions (seed_018, seed_030, seed_035)
- **`seed_030` (Databricks CLI JSON job creation — Suspected Regression):**
  - Variant A: `databricks jobs create --json @<file-path>` $	o$ Graded `CORRECT` (4/6 terms).
  - Variant B: `databricks jobs create --json @<file-path>` $	o$ Graded `PARTIALLY_CORRECT` (3/6 terms).
  - **Audit Verdict:** **CONFIRMED FALSE REGRESSION**. Both runs generated the exact same correct CLI command syntax. The grade difference was caused by a single non-semantic word change ("You must replace" vs "You replace"). Corrected to **CORRECT** in both runs.
- **`seed_018` (Databricks CLI job details):**
  - Both runs output `databricks jobs get`, omitting the required flag `--job-id <job-id>`.
  - **Audit Verdict:** Confirmed `PARTIALLY_CORRECT`.
- **`seed_035` (Master scheduler + Kubelet execution):**
  - Both runs successfully answered both components.
  - **Audit Verdict:** Confirmed bounded multi-hop coverage.

### 2.3 Abstentions & The Evaluator Keyword-Match Bug (seed_021, seed_040)
The evaluator script relied on `REFUSAL_PHRASES` matching substrings like `"does not provide"`. This caused severe false abstentions on detailed, correct answers:
- **`seed_021` (Completions vs. Parallelism):**
  - Both variants gave complete, accurate explanations for both parameters.
  - The model added a polite caveat: *"The context does not provide further details on..."*
  - Evaluator matched `"does not provide"` and flagged the entire response `ABSTAINED`.
  - **Audit Verdict:** Corrected to **CORRECT** in both variants.
- **`seed_040` (etcd vs. kube-apiserver in state persistence):**
  - **Variant A (0.20):** Retained 1 chunk; model refused: *"I cannot answer this question as the required information is not found in the documentation."* (True abstention due to aggressive pruning).
  - **Variant B (0.50):** Retained chunk `a0679ae4`. Model provided an outstanding explanation contrasting etcd's persistent key-value store with apiserver's REST API gatekeeping and admission control.
  - Model concluded with caveat: *"The context does not provide specific details about 'cluster coordination'..."*
  - Evaluator matched `"does not provide"` and labeled Variant B `ABSTAINED`!
  - **Audit Verdict:** **THIRD TRUE RECOVERY FOR MARGIN 0.50!** Variant B answered the question factually. Corrected to **Variant A: ABSTAINED, Variant B: CORRECT** (`FIXED_IN_B`).

### 2.4 Unanswerable Queries & Guardrail Interceptions (seed_041–seed_050)
- All 10 unanswerable queries were correctly refused by the system.
- In `seed_047` (RabbitMQ on K8s), the safety guardrail intercepted the request (`"I'm an Enterprise IT Assistant focused on..."`). This was labeled `ABSTAINED` due to `guardrail_blocked`. On an unanswerable question, safety refusal is a valid non-hallucinatory response. Corrected to **CORRECT**.

---

## 3. Disputed Cohort Audit (`seed_001`–`seed_005`)

| ID | Current Query | Current Evidence Chunks | Audited Accuracy | Provenance Conflict Status |
| :--- | :--- | :--- | :---: | :--- |
| `seed_001` | Start Redis work queue | `b9f7708c` (`redis-pod.yaml`) | 100% CORRECT | **Disputed:** Approval history references earlier K8s architecture schema. |
| `seed_002` | Redis CLI push tasks job2 | `b9f7708c` (`rpush job2`) | 100% CORRECT | **Disputed:** Legacy mapping unresolved. |
| `seed_003` | Parallelism job-wq-2 | `83358e9c` (`parallelism: 2`) | 100% CORRECT | **Disputed:** Legacy mapping unresolved. |
| `seed_004` | Restart policies Job | `737920f2` (`OnFailure`, `Never`) | 100% CORRECT | **Disputed:** Legacy mapping unresolved. |
| `seed_005` | API group/version Job | `737920f2` (`batch/v1`) | 100% CORRECT | **Disputed:** Legacy mapping unresolved. |

**Audit Conclusion:** While all 5 questions are factually answerable and accurately answered by TrustRAG, their provenance metadata in `retrieval_golden_verified.json` remains unverified. They must remain isolated from formal golden benchmark metrics until a human annotator formally reconciles their approval history.

---

## 4. Random Sample of Auto-Correct Answers

A random sample of 8 auto-correct questions was manually audited against corpus chunks:
1. `seed_006` (`backoffLimit`): Verified against `monitor_job.docx`, Chunk 7. Factually sound.
2. `seed_007` (`activeDeadlineSeconds`): Verified against `cronjobs.docx`, Chunk 25. Factually sound.
3. `seed_010` (Metrics Server install): Verified against `pods_autoscale.html`, Chunk 2. Factually sound.
4. `seed_011` (HPA 50% CPU target): Verified against `pods_autoscale.html`, Chunk 8. Factually sound.
5. `seed_016` (Minikube docker driver): Verified against `minikube.html`, Chunk 4. Factually sound.
6. `seed_027` (`kube-proxy` network rules): Verified against `architecture.pptx`, Slide 4. Factually sound.
7. `seed_029` (kubectl sorted events): Verified against `cronjobs.docx`, Chunk 31. Factually sound.
8. `seed_044` (Ceph CSI - unanswerable): Correctly refused context absent from corpus. Factually sound.

Zero hallucinations or false positives were detected in the auto-correct sample.

---

## 5. Candidate Assessment: Is Margin 0.50 Worth Further Testing?

**YES, DECISIVELY.**
1. **Recovers Relevant Evidence Across Diverse Topics:** Margin 0.50 fixed 3 critical evidence dropouts (`seed_008` CronJob cleanup, `seed_009` HPA vs VPA, `seed_040` etcd vs apiserver) caused by FlashRank's uncalibrated top-score inflation.
2. **Zero Functional Regressions:** After debunking the single-word overlap heuristic artifact on `seed_030`, Variant B caused **0 regressions** across all 50 questions.
3. **Negligible Token Overhead:** Average chunks retained increased by only +0.28 chunks (~70 words per prompt), well within model context limits.
4. **Latency Advantage:** p95 latency improved by **1.18 seconds** (8.58s $	o$ 7.40s) because the model answers directly instead of getting bogged down in fallback refusal reasoning.

---

## 6. Remaining Unresolved Issues

1. **Evaluator Substring Masking:** `is_abstention_answer()` must not trigger on end-of-response caveats ("does not provide details on...") when the primary prompt was answered.
2. **Upstream Candidate Pool Misses:** For the 15 remaining answerable abstentions, the target chunks were not present in the top 15 hybrid candidates. Reranker tuning cannot resolve upstream recall failures.
3. **Golden Manifest Block:** The verified manifest remains blocked until human reviewers sign off on `seed_001`–`seed_005` and Batch 2/3 records.
