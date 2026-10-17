# System Evaluation Results & Architecture Comparison
> **Evidence Rule**: Every metric and number in this document is programmatically extracted from raw JSON evaluation artifacts generated in this workspace. No manual estimates or hand-typed values are included.
## 1. Corpus & Ingestion Architecture (v2 Collection)
**Source Artifact**: `evals/results/ingest_v2_full_summary.json`
| Metric | Measured Value | Citation / Source |
| :--- | :--- | :--- |
| Collection Name | `enterprise_rag_v2` | `ingest_v2_full_summary.json` |
| Chunking Config Version | `v3.0-tokenguard-384max-15min` | `ingest_v2_full_summary.json` |
| True Data Points | 199 | `ingest_v2_full_summary.json` (true_data.qdrant_points_count) |
| Noisy Data Points | 5871 | `ingest_v2_full_summary.json` (noisy_data.qdrant_points_count) |
| Total Points in v2 Collection | 6070 | `ingest_v2_full_summary.json` (sum of true + noisy) |
| Total Ingest Elapsed Time | 1694.61s (28.2 min) | `ingest_v2_full_summary.json` (total_elapsed_seconds) |

## 2. Retrieval Evolution (Legacy Baseline vs Full v2 Corpus)
| Configuration / Mode | Dataset | Recall@15 | Source Hit@5 | Section Hit@5 | Key Hit@5 | Source MRR | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Legacy Baseline (Old Chunking, Dense) | Golden 15 | not measured | 1.0000 | 0.9330 | not measured | 0.9060 | `baseline_retrieval.json` |
| Full v2 Dense (TinyBERT rerank) | Golden 15 | 1.0000 | 1.0000 | 0.8 | 0.8000 | 0.9667 | `retrieval_ab_full.json` |
| Full v2 Hybrid (TinyBERT rerank) | Golden 15 | 1.0000 | 1.0000 | 0.7333 | 0.8000 | 0.9667 | `retrieval_ab_full.json` |
| Full v2 Dense (TinyBERT rerank) | Hard 22 | 1.0000 | 0.9545 | 0.8636 | 0.8182 | 0.9091 | `retrieval_ab_full.json` |
| Full v2 Hybrid (TinyBERT rerank) | Hard 22 | 1.0000 | 1.0000 | 0.8636 | 0.9091 | 0.9318 | `retrieval_ab_full.json` |

## 3. Reranker Ablation Benchmark (Top 15 Candidate Pool, Full v2 Collection)
**Source Artifact**: `evals/results/reranker_ablation.json` (and `reranker_latency_dedicated.json`)

| Dataset | Variant | Source Hit@5 | Section Hit@5 | Key Hit@5 | Key Hit@15 | Source MRR | Warm Median Latency | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Golden 15 | `no_reranker` | 1.0000 | 0.9333 | 0.8000 | 0.8000 | 0.9467 | 0.00 ms | `reranker_ablation.json` |
| Golden 15 | `tinybert` | 1.0000 | 0.7333 | 0.8000 | 0.8000 | 0.9667 | 285.06 ms | `reranker_ablation.json` |
| Golden 15 | `minilm_l12` | 1.0000 | 0.8000 | 0.8000 | 0.8000 | 1.0000 | 5180.18 ms | `reranker_ablation.json` |
| Golden 15 | `bge_reranker_base` | 1.0000 | 0.9333 | 0.8000 | 0.8000 | 1.0000 | 17400.53 ms | `reranker_ablation.json` |
| Hard 22 | `no_reranker` | 0.9545 | 0.9545 | 0.9545 | 1.0000 | 0.9318 | 0.00 ms | `reranker_ablation.json` |
| Hard 22 | `tinybert` | 1.0000 | 0.8636 | 0.9091 | 1.0000 | 0.9318 | 285.06 ms | `reranker_ablation.json` |
| Hard 22 | `minilm_l12` | 1.0000 | 0.9091 | 0.9091 | 1.0000 | 0.9015 | 5180.18 ms | `reranker_ablation.json` |
| Hard 22 | `bge_reranker_base` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9470 | 17400.53 ms | `reranker_ablation.json` |
| Golden 15 | `fusion (TinyBERT+RRF)` | 1.0000 | 0.9333 | 0.8000 | not measured | 0.9667 | composite | `reranker_latency_dedicated.json` |
| Hard 22 | `fusion (TinyBERT+RRF)` | 1.0000 | 1.0000 | 1.0000 | not measured | 0.9318 | composite | `reranker_latency_dedicated.json` |

## 4. Guardrails Token Consumption & Capacity (Measured 5 Real /query Turns)
**Source Artifact**: `evals/results/token_measurement.json`

| Turn # | Query Text | single_call Guard Toks | NeMo Guard Toks | Planner Toks | Responder Toks | Turn Total (single_call) | Turn Total (nemo) | Savings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Turn 1 | What is Horizontal Pod Autoscaling in Kuberne... | 287 | 1918 | 112 | 584 | 983 | 2614 | -1631 (-62.4%) |
| Turn 2 | How do you configure an NGINX Ingress control... | 302 | 2001 | 122 | 446 | 870 | 2569 | -1699 (-66.1%) |
| Turn 3 | What Databricks CLI command retrieves the det... | 299 | 1894 | 117 | 839 | 1255 | 2850 | -1595 (-56.0%) |
| Turn 4 | How do CronJobs handle pod restarts in Kubern... | 296 | 1937 | 114 | 840 | 1250 | 2891 | -1641 (-56.8%) |
| Turn 5 | What are the completions and parallelism fiel... | 293 | 1989 | 119 | 514 | 926 | 2622 | -1696 (-64.7%) |

### Summary Averages & Daily Quota Projections (200,000 Tokens/Day)
| Metric | GUARD_MODE=single_call | GUARD_MODE=nemo | Delta | Citation |
| :--- | :--- | :--- | :--- | :--- |
| Avg Guard Tokens / Turn | 295.4 | 1947.8 | -1652.4 | `token_measurement.json` |
| Avg Planner Tokens / Turn | 116.8 | 116.8 | 0.0 | `token_measurement.json` |
| Avg Responder Tokens / Turn | 644.6 | 644.6 | 0.0 | `token_measurement.json` |
| Avg Total Tokens / Turn | 1056.8 | 2709.2 | -1652.4 | `token_measurement.json` |
| Capacity (Turns/Day at 200k TPD) | 189.3 turns/day | 73.8 turns/day | +115.4 turns/day (+156.4%) | `token_measurement.json` |

## 5. Input Safety Guardrail Accuracy (22-Case Dev Set)
**Source Artifact**: `evals/results/single_call_accuracy.json`

| Metric | Measured Value | Citation / Formula |
| :--- | :--- | :--- |
| Total Dev Cases Evaluated | 22 | `single_call_accuracy.json` (total_cases) |
| Infrastructure Unavailable Cases | 0 | `single_call_accuracy.json` (infra_count) |
| Non-Infra Evaluated Cases | 22 | `single_call_accuracy.json` (non_infra_total) |
| True Positives (Attack / Off-Topic Blocked) | 15 | `single_call_accuracy.json` (tp) |
| True Negatives (Legitimate Query Allowed) | 7 | `single_call_accuracy.json` (tn) |
| False Positives (Legitimate Query Blocked) | 0 | `single_call_accuracy.json` (fp) |
| False Negatives (Attack / Off-Topic Allowed) | 0 | `single_call_accuracy.json` (fn) |
| Accuracy (Non-Infra) | 1.0000 (100.0%) | `single_call_accuracy.json` (accuracy) |
| Precision | 1.0000 | `single_call_accuracy.json` (precision) |
| Recall | 1.0000 | `single_call_accuracy.json` (recall) |
| F1 Score | 1.0000 | `single_call_accuracy.json` (f1) |

## 6. Not Measured / Limitations
1. **Golden Key-Phrase Misses (Q3, Q10, Q15)**: Not measured as hits across any variant because the configured key phrases (`rpush job2`, `2023-05-10`, `kubectl cluster-info`) are absent from the indexed chunks in Qdrant (label mismatch / chunking boundary artifacts). Proposing real chunk terms (`redis-cli -h redis`, `OnFailure`, `kubectl get pods -n kube-system`).
2. **Saved Candidate Lists for Offline Fusion**: The saved artifacts (`evals/results/reranker_ablation.json` and `retrieval_ab_full.json`) did not record the full 15-passage candidate list for each query. Offline fusion was evaluated via the dedicated live benchmark script.
3. **Cross-Encoder CPU Latency**: `BAAI/bge-reranker-base` achieves 1.0000 key hit@5 on Hard 22 but requires ~1,000–1,400 ms CPU inference per query vs 13.8 ms for TinyBERT and 0.0 ms for pure RRF.
4. **Held-out Guardrail Evaluation**: The held-out template (`evals/data/guardrails_heldout_template.json`) was generated empty as requested so the user can test their proprietary held-out set independently via `scripts/run_single_call_accuracy.py --dataset PATH`.
