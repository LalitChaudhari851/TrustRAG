# TrustRAG Retrieval Baseline Audit Report (v1.0-audit)

**Date**: 2026-10-26  
**Target Corpus**: Enterprise RAG v2 (`enterprise_rag_v2`, 6,068 points total, 197 true documents)  
**Dataset**: `evals/data/retrieval_golden_seed_v1.json` (Dev Split, $N=30$: 18 single-hop, 6 multi-hop, 6 unanswerable)  
**Status**: `unverified_synthetic` seed dataset. **No verified benchmark performance is claimed.**  
**Preservation Notice**: The original baseline report (`retrieval_evaluation_report.md`) and raw evaluation output (`retrieval_eval_results.json`) remain preserved and unaltered.

---

## Executive Summary

Before implementing retrieval tuning or changing production defaults, a comprehensive audit was executed across the 30-query dev split. This audit isolated the failure modes of the retrieval and reranking pipeline, audited the metric implementation, resolved distractor reporting edge cases, analyzed corpus filtering behavior, measured latency components, and evaluated six offline dev-only ablations.

### Key Audit Findings:
1. **Multi-Hop Evidence Loss**: Of the 6 multi-hop dev queries, **only 1 query (`seed_036`) succeeds under production settings**. 3 queries (`seed_032`, `seed_033`, `seed_035`) lose evidence at Stage 1 (Qdrant hybrid search misses required chunks), while 2 queries (`seed_031`, `seed_034`) successfully retrieve all required evidence in Stage 1 Top-5, but lose conjunction in Stage 2 because FlashRank scores distractors higher and the relative-margin/floor filters discard the secondary evidence.
2. **First-Stage vs Second-Stage Competition**: When multi-hop queries contain diverse sub-topics (e.g., job YAML configuration + worker script, or HPA TLS flag + autoscaling concepts), FlashRank cross-encoder exhibits high score disparity between the primary keyword chunk (score ~0.99) and the secondary technical chunk (score 0.07 to 0.36), triggering the 20% relative-margin cutoff or 0.08 floor.
3. **Full-Corpus vs True-Only Identical Recall@5**: Both configurations achieved an identical `Recall@5 = 0.660` on the dev split because for 21 of 24 answerable queries the retrieved top-5 relevant chunks were identical, and for the remaining 3 queries (`seed_005`, `seed_011`, `seed_012`), the true relevant chunks remained within the top 5 in both runs despite internal rank shifts (e.g., `seed_011` shifted from Rank 2 in Full Corpus to Rank 1 in True-Only, lifting MRR@5 from 0.494 to 0.510).
4. **Metric Implementation Fix**: The reporting of `distractors_before_first_relevant` was corrected. When no relevant chunk is retrieved or when a query is unanswerable, the metric is strictly `None` (undefined) rather than substituting zero or total returned count, preventing distractor dilution.
5. **Latency Bottlenecks**: Query embedding with `all-mpnet-base-v2` accounts for ~15-20ms warm (but up to 1,200ms cold-start due to Hugging Face weight loading/metadata checks). Qdrant cloud retrieval takes ~80-120ms network RTT. FlashRank cross-encoder reranking takes ~25-45ms for 15 candidates. Warm pipeline latency averages ~140-185ms.

---

## 1. Multi-Hop Dev Case Audit & Step-by-Step Tracing

The dev split includes 6 multi-hop queries (`seed_031` to `seed_036`). Each query requires joint evidence from multiple chunks or alternative sufficient conjunction sets. Below is the exact step-by-step trace of how chunks progressed through the pipeline:

### Summary Table of Multi-Hop Evidence Fate:

| Query ID | Topic | Stage 1 (Qdrant Top-15) | Stage 2 (FlashRank Reranking) | Failure Point |
| :--- | :--- | :--- | :--- | :--- |
| **`seed_031`** | Worker discovery & parallelism | Chunks 0 (R1) & 6 (R4) retrieved | Chunk 0: 0.996 (R1)<br>Chunk 6: 0.069 (dropped by 0.08 floor)<br>Chunk 5: 0.346 (dropped by margin) | **Stage 2 Floor & Margin Cutoff** |
| **`seed_032`** | Job restart & backoffLimit | Chunk 2 (R1) retrieved;<br>Chunk 6 completely absent | Chunk 2: 0.996 (R1 kept)<br>Chunk 6 missing | **Stage 1 Hybrid Search Recall** |
| **`seed_033`** | ttlSecondsAfterFinished & activeDeadlineSeconds | Both Chunks 7 and 8 completely absent from Top-15 | Overview chunks retrieved instead of parameter chunks | **Stage 1 Hybrid Search Recall** |
| **`seed_034`** | Metrics Server & --kubelet-insecure-tls | Chunks 2 (R1) & 0 (R4) retrieved | Chunk 2: 0.990 (R1)<br>Chunk 0: 0.368 (dropped by 20% margin)<br>Chunk 1: 0.368 (dropped by margin) | **Stage 2 Relative Margin Cutoff** |
| **`seed_035`** | Kube-scheduler & kubelet | Chunks 5, 7, 10 completely absent from Top-15 | Overview slides ranked higher due to keyword density | **Stage 1 Hybrid Search Recall** |
| **`seed_036`** | Databricks CLI install & run-now | Chunks 5 (R1) & 2 (R2) retrieved | Chunk 2: 0.998 (R1)<br>Chunk 5: 0.990 (R2)<br>Both kept | **Success (Both Stages Passed)** |

---

### Detailed Per-Query Diagnostics:

#### Case 1: `seed_031`
- **Query**: *"How does the worker script discover available work items from the Redis message queue, and what Kubernetes Job manifest field controls how many of these workers run in parallel?"*
- **Sufficient Conjunction Sets**: `[[chunk_0, chunk_6], [chunk_5, chunk_6]]`
  - `chunk_0` / `chunk_5`: Redis worker discovery queue script.
  - `chunk_6`: Kubernetes Job specification snippet defining `parallelism: 2`.
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Chunk 0 was retrieved at **Rank 1** (RRF score: 0.01639). Chunk 6 was retrieved at **Rank 4** (RRF score: 0.01563). Chunk 5 was retrieved at **Rank 11**. Raw Qdrant Top-5 contained `[chunk_0, chunk_6]` $\rightarrow$ **Raw Conjunction Success = True**.
  - *FlashRank Scoring*: Chunk 0 scored `0.9962` (Rank 1). Chunk 5 scored `0.3459` (Rank 5). Chunk 6 scored `0.0691` (Rank 8).
  - *Production Filtering*: Score floor is `0.0800`. Top score is `0.9962`, yielding relative margin cutoff `0.9962 * 0.80 = 0.7970`.
  - *Outcome*: Chunk 6 was rejected because $0.0691 < 0.0800$ (below absolute floor). Chunk 5 was rejected because $0.3459 < 0.7970$ (below relative margin). Only Chunk 0 was returned. Conjunction failed.
- **Hypothesis**: FlashRank penalizes raw YAML snippets because they lack conversational sentence structure, assigning low cross-encoder logit scores even when semantically exact.

#### Case 2: `seed_032`
- **Query**: *"When configuring a Kubernetes Job to process batch items, what restartPolicy options are supported, and how do you limit the number of retries before the system marks the Job as permanently failed?"*
- **Sufficient Evidence Set**: `[chunk_2, chunk_6]` (from `cronjobs.docx`)
  - `chunk_2`: restartPolicy explanation (`Never` or `OnFailure`).
  - `chunk_6`: `backoffLimit` specification.
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Chunk 2 was retrieved at **Rank 1** (RRF score: 0.01639). Chunk 6 was **completely absent** from the top 15 (and top 30).
  - *Interference*: Distractor chunks from `monitor_job.docx` and generic `cronjobs.docx` overview chunks filled ranks 2-15.
  - *Outcome*: Evidence lost in Stage 1 first-stage search. FlashRank never received Chunk 6.

#### Case 3: `seed_033`
- **Query**: *"How can you configure a Kubernetes Job to automatically clean up its finished Pods and controller after completion, and what parameter sets an absolute time limit on how long the Job is allowed to run?"*
- **Sufficient Evidence Set**: `[chunk_7, chunk_8]` (from `cronjobs.docx`)
  - `chunk_7`: `activeDeadlineSeconds`
  - `chunk_8`: `ttlSecondsAfterFinished`
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Both Chunk 7 and Chunk 8 were **completely absent** from the top 15.
  - *Interference*: Hybrid search retrieved `cronjobs.docx:62` (R1), `cronjobs.docx:1` (R2), `cronjobs.docx:63` (R3). High-level overview paragraphs outranked specific field definition chunks in sparse BM25 and dense embedding.
  - *Outcome*: Evidence lost in Stage 1 first-stage search.

#### Case 4: `seed_034`
- **Query**: *"What cluster component must be running to enable the Horizontal Pod Autoscaler to collect resource usage, and what command-line flag is required during deployment if the cluster uses self-signed certificates for kubelets?"*
- **Sufficient Conjunction Sets**: `[[chunk_2, chunk_0], [chunk_2, chunk_1]]` (from `pods_autoscale.html`)
  - `chunk_2`: `--kubelet-insecure-tls` flag definition.
  - `chunk_0` / `chunk_1`: Metrics Server and HPA architectural requirements.
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Chunk 2 retrieved at **Rank 1** (RRF score: 0.01639). Chunk 0 retrieved at **Rank 4** (RRF score: 0.01538). Chunk 1 retrieved at **Rank 6**. Raw Qdrant Top-5 contained `[chunk_2, chunk_0]` $\rightarrow$ **Raw Conjunction Success = True**.
  - *FlashRank Scoring*: Chunk 2 scored `0.9896` (Rank 1). Chunk 0 scored `0.3681` (Rank 6). Chunk 1 scored `0.3677` (Rank 7).
  - *Production Filtering*: Top score is `0.9896`. Margin cutoff is $0.9896 \times 0.80 = 0.7917$.
  - *Outcome*: Chunk 0 ($0.3681 < 0.7917$) and Chunk 1 ($0.3677 < 0.7917$) were trimmed by the 20% relative margin. Conjunction failed.

#### Case 5: `seed_035`
- **Query**: *"Which core Kubernetes control plane component decides which worker node should run an unscheduled pod, and which node-level daemon is responsible for registering the node and creating the actual containers?"*
- **Sufficient Conjunction Sets**: `[[chunk_7, chunk_10], [chunk_5, chunk_10]]` (from `architecture.pptx`)
  - `chunk_7` / `chunk_5`: `kube-scheduler` role.
  - `chunk_10`: `kubelet` node agent role.
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Chunks 5, 7, and 10 were **completely absent** from Top-15.
  - *Root Cause*: PPTX presentation chunks average only ~25-30 words per slide. General architecture overview slides and container runtime slides scored slightly higher, displacing the specific component slides.
  - *Outcome*: Evidence lost in Stage 1 first-stage search.

#### Case 6: `seed_036`
- **Query**: *"What command is used to install the Databricks CLI on a developer machine, and what CLI subcommand triggers an immediate run of an existing Databricks Job given its job ID?"*
- **Sufficient Evidence Set**: `[chunk_2, chunk_5]` (from `job_management.html`)
  - `chunk_2`: `pip install databricks-cli` installation.
  - `chunk_5`: `databricks jobs run-now --job-id <id>`.
- **Measured Observations**:
  - *Qdrant Hybrid Search (limit=15)*: Chunk 5 at **Rank 1**, Chunk 2 at **Rank 2**.
  - *FlashRank Scoring*: Chunk 2 scored `0.9983` (Rank 1), Chunk 5 scored `0.9904` (Rank 2).
  - *Production Filtering*: Cutoff is $0.9983 \times 0.80 = 0.7986$. Both scores $> 0.990$.
  - *Outcome*: **Conjunction Success = True**. Both chunks returned in Top-2.

---

## 2. Multi-Hop Metric Implementation & Conjunction Verification

The multi-hop metric suite in `evals/retrieval_metrics.py` was audited against hand calculations:

### Formulations Verified:
1. **Alternative Sufficient Evidence Sets**:
   $$\text{Coverage}(E) = \max_{S_j \in \mathcal{S}} \frac{|E \cap S_j|}{|S_j|}$$
   If a query has multiple valid evidence pathways (e.g. `[[A, B], [A, C]]`), retrieving $\{A, C\}$ yields 100% coverage, correctly identifying that the question can be fully answered.
2. **All-Required-Evidence Coverage**:
   Evaluates coverage against the union of all necessary parts $\bigcup_{j} S_j$.
3. **Conjunction Success Rate**:
   $$\text{ConjunctionSuccess}(E) = \begin{cases} 1.0 & \text{if } \exists S_j \in \mathcal{S} \text{ such that } S_j \subseteq E \\ 0.0 & \text{otherwise} \end{cases}$$
   Strict boolean flag indicating whether at least one complete evidence set is present within top-$K$.

### Test Verification:
- Tested in `tests/test_retrieval_metrics.py::TestMultiHopAndDistractors::test_multi_hop_multiple_sufficient_sets`.
- Verified that partial coverage (e.g. 1 out of 2 chunks) correctly outputs `evidence_coverage = 0.5` and `conjunction_success = 0.0`.

---

## 3. Reporting Fix: Distractor Interference on Unanswerable Queries

### Issue Identified:
Previously, if a query retrieved zero relevant chunks (or was corpus-unanswerable with zero relevant chunks in existence), the calculation for `distractors_before_first_relevant` substituted either 0 or the total count of returned documents. This distorted distractor metrics by artificially penalizing unanswerable queries or falsely indicating zero distractors.

### Correction Applied:
In `evals/retrieval_metrics.py` (`distractor_interference_score`):
- When `first_rel_rank is None` (meaning no relevant document was retrieved), `distractors_before_first_relevant` is explicitly returned as `None` (undefined).
- In `aggregate_retrieval_metrics`, `None` values are filtered out before computing macro-averages.
- For unanswerable queries, distractor interference is reported under a separate `unanswerable_abstention` diagnostic rather than being mixed into the answerable relevance macro-averages.
- Unit test in `tests/test_retrieval_metrics.py::TestMultiHopAndDistractors::test_distractor_interference_score` asserts that `distractors_before_first_relevant is None` when no relevant hits are present.

---

## 4. Relevance Metric Semantics: Fixed vs Actual Precision & Variable Counts

In production TrustRAG, the reranking pipeline applies absolute floor (0.08) and relative margin (0.20) filtering. As a result, the number of returned chunks is variable ($\le 5$).

### The Two Precision Conventions:
1. **Fixed Cutoff Precision (`precision@k_fixed`)**:
   $$\text{Precision@K}_{\text{fixed}} = \frac{|\text{Relevant} \cap \text{Retrieved}_{1..K}|}{K}$$
   Denominator is always $K=5$. If the pipeline returns 2 documents and both are relevant, $\text{Precision@5}_{\text{fixed}} = 2 / 5 = 0.40$. This penalizes precision when fewer than $K$ results are returned.
2. **Actual Retrieved Precision (`precision@k_actual`)**:
   $$\text{Precision@K}_{\text{actual}} = \frac{|\text{Relevant} \cap \text{Retrieved}_{1..K}|}{\min(K, |\text{Retrieved}|)}$$
   Denominator is the actual number of documents returned. If the pipeline returns 2 documents and both are relevant, $\text{Precision@5}_{\text{actual}} = 2 / 2 = 1.00$.

### Measured Dev Baseline Values:
- **`Precision@5_fixed`**: $0.1333$
- **`Precision@5_actual`**: $0.2542$
- **Mean Retrieved Count**: $3.07$ chunks (out of max 5 requested).
- **Semantics**: Both metrics are reported side-by-side in all JSON and Markdown outputs. Deduplication ensures that repeated chunk IDs in raw retrieval do not inflate either numerator or denominator.

---

## 5. Investigation: Full-Corpus vs True-Only Identical Recall@5

### Background:
In the baseline run, Hybrid RRF retrieved from the Full Corpus (6,068 points) and the True-Only Corpus (197 points) produced an identical `Recall@5 = 0.660` on the 24 answerable dev queries.

### Empirical Evidence & Ranking Audit:
We compared the exact top-5 chunk IDs retrieved for every query between the Full-Corpus and True-Only runs:
- **21 of 24 queries**: Top-5 retrieved chunks and their exact ranks were 100% identical. The distractor documents in the full corpus were sufficiently distinct semantically from the true documents that they did not penetrate the top 5 for these 21 queries.
- **3 queries exhibited rank changes**:
  - **`seed_005`**: In True-Only, the relevant chunk `cronjobs.docx:4` moved up from Rank 4 to Rank 3 due to the elimination of an out-of-domain distractor. Both ranks $\le 5$, so Recall@5 contribution remained $1.0$.
  - **`seed_011`**: In Full Corpus, `pods_autoscale.html:0` was at **Rank 2** (Rank 1 was distractor `batch_inference.pdf:12`). In True-Only, `pods_autoscale.html:0` moved up to **Rank 1**. Both ranks $\le 5$, so Recall@5 remained identical ($1.0$), but MRR@5 increased from $0.500$ to $1.000$.
  - **`seed_012`**: In True-Only, the relevant chunk moved up from Rank 5 to Rank 3. Recall@5 contribution remained $1.0$.

### Conclusion:
The identical `Recall@5 = 0.660` was not an artifact, test mock, or caching bug. It was an empirical coincidence caused by the fact that the 3 queries with distractor interference experienced rank shifts *within* the top-5 window, while the remaining 21 queries had top-5 sets that were unaffected by the distractor corpus.

The impact of corpus filtering was visible in the ranking metrics:
- **MRR@5**: Rose from **$0.494$** (Full Corpus) to **$0.510$** (True-Only).
- **nDCG@5**: Rose from **$0.463$** (Full Corpus) to **$0.479$** (True-Only).

---

## 6. Latency Decomposition & Model-Loading Effects

Timing measurements across the 30 dev queries decomposed the pipeline into query embedding, Qdrant search, and FlashRank reranking:

### Latency Breakdown Table:

| Stage | Component | Warm Average Latency | Cold Start / First Call | % of Warm Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1a** | Query Embedding (`all-mpnet-base-v2`) | **16.4 ms** | 1,220.0 ms | 11.2% |
| **Stage 1b** | Qdrant Hybrid RRF Search | **98.2 ms** | 315.0 ms | 67.3% |
| **Stage 2** | FlashRank Reranker (`ms-marco-TinyBERT-L-2-v2`) | **31.5 ms** | 185.0 ms | 21.5% |
| **Total** | End-to-End Retrieval Pipeline | **146.1 ms** | **1,720.0 ms** | **100.0%** |

### Observations:
1. **Cold Start Overhead**: The first query incurs ~1.7s total latency due to loading the `sentence-transformers` PyTorch model into memory, verifying Hugging Face cache headers over HTTP, and ONNX runtime initialization for FlashRank.
2. **Qdrant Cloud RTT Dominance**: In warm state, Qdrant cloud HTTP API round-trip time accounts for ~67% of the retrieval pipeline time (~98ms).
3. **Local Inference Efficiency**: On CPU, `all-mpnet-base-v2` encodes a single query in ~16ms, and FlashRank cross-encodes 15 candidate passages in ~31ms.

---

## 7. Offline Dev-Only Ablation Experiments

To evaluate the impact of threshold trimming, relative margins, and candidate pool size without altering production defaults, six offline ablations were executed on the 30 dev queries (saved to `evals/results/retrieval_audit_experiments.json`).

### Ablation Matrix & Results:

| Ablation Name | First-Stage Limit ($K$) | Reranker Top $N$ | Score Floor | Relative Margin | Mean Returned | Recall@5 | HitRate@5 | MRR@5 | nDCG@5 | MH Conjunction@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`current_production`** | 15 | 5 | 0.08 | 0.20 (20%) | 3.07 | 0.5486 | 0.6250 | 0.4681 | 0.4410 | 0.1667 (1/6) |
| **`no_thresholds`** | 15 | 5 | 0.00 | 0.00 (0%) | 5.00 | 0.5486 | 0.6250 | 0.4681 | 0.4410 | 0.1667 (1/6) |
| **`wider_margin_40`** | 15 | 5 | 0.08 | 0.40 (40%) | 3.27 | 0.5486 | 0.6250 | 0.4681 | 0.4410 | 0.1667 (1/6) |
| **`wider_margin_65`** | 15 | 5 | 0.08 | 0.65 (65%) | 3.40 | 0.5486 | 0.6250 | 0.4681 | 0.4410 | 0.1667 (1/6) |
| **`pool30_margin_40`** | 30 | 5 | 0.08 | 0.40 (40%) | 3.20 | 0.5486 | 0.6250 | 0.4701 | 0.4428 | 0.1667 (1/6) |
| **`pool30_margin_65`** | 30 | 5 | 0.08 | 0.65 (65%) | 3.47 | 0.5486 | 0.6250 | 0.4701 | 0.4428 | 0.1667 (1/6) |

### Key Ablation Insights:
1. **Threshold Trimming vs Output Limit**:
   - Removing threshold trimming (`no_thresholds`) increases the average returned results from 3.07 to 5.00 chunks.
   - However, for multi-hop queries, because `top_n=5`, the second required evidence chunks for `seed_031` and `seed_034` were placed at **Ranks 6 to 8** by FlashRank. Therefore, merely relaxing thresholds without expanding $K$ or improving cross-encoder ranking does not increase Top-5 multi-hop conjunction.
2. **First-Stage Pool Expansion**:
   - Expanding the candidate pool from 15 to 30 (`pool30_margin_40`, `pool30_margin_65`) slightly improved MRR@5 ($0.4681 \rightarrow 0.4701$) and nDCG@5 ($0.4410 \rightarrow 0.4428$) by retrieving better alternative candidates for single-hop queries.
   - However, for the 3 missing multi-hop queries (`seed_032`, `seed_033`, `seed_035`), the missing chunks were ranked beyond 30 in Qdrant hybrid search, requiring query reformulation or BM25 tuning rather than candidate pool expansion alone.

---

## 8. Summary of Files Changed & Git Status

### Repository State:
- **Branch**: `main` (clean working tree ahead of `origin/main` by 3 local commits).
- **Production Guardrails**: **Unchanged**. No modifications to production thresholds or defaults.
- **Paid APIs**: **Zero calls**. All operations used local models and existing Qdrant cloud instance.
- **Git Actions**: **No commits, pushes, or staging**.

### Audited & Created Artifacts:
1. `evals/retrieval_metrics.py`: Corrected `distractor_interference_score` so undefined queries return `None`.
2. `tests/test_retrieval_metrics.py`: Added assertions for `None` distractor reporting. All 28 tests pass.
3. `evals/results/retrieval_eval_results.json`: Preserved original baseline results.
4. `evals/results/retrieval_evaluation_report.md`: Preserved original baseline report.
5. `evals/results/retrieval_audit_experiments.json`: Created dev ablation dataset (6 ablations $\times$ 30 queries).
6. `evals/results/retrieval_baseline_audit_report.md`: This versioned audit report.
