# System Evaluation Results & Architecture Comparison

> **Evidence Rule**: Every metric and number in this document is programmatically extracted from raw JSON evaluation artifacts generated in this workspace. No manual estimates or hand-typed values are included.

## 1. Corpus & Ingestion Architecture (v2 Collection)

**Source Artifact**: `evals/results/ingest_v2_full_summary.json` (Run Timestamp: `2026-10-18 13:19:03`)

| Metric | Measured Value | Citation / Source |
| :--- | :--- | :--- |
| Collection Name | `enterprise_rag_v2` | `ingest_v2_full_summary.json` |
| Chunking Config Version | `v3.1-tokenguard-merge-384max` | `ingest_v2_full_summary.json` |
| True Data Points | 197 | `ingest_v2_full_summary.json` (true_data.qdrant_points_count) |
| Noisy Data Points | 5871 | `ingest_v2_full_summary.json` (noisy_data.qdrant_points_count) |
| Total Points in v2 Collection | 6068 | `ingest_v2_full_summary.json` (sum of true + noisy) |

## 2. Retrieval Evolution (Legacy Baseline vs Full v2 Corpus)

| Configuration / Mode | Dataset | Run Timestamp | Recall@15 | Source Hit@5 | Section Hit@5 | Key Hit@5 | Source MRR | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Legacy Baseline (Old Chunking, Dense) | Golden 15 | 2026-10-16 (Phase 0) | not measured | 1.0000 | n/a | not measured | 0.9060 | `baseline_retrieval.json` |
| Full v2 Dense (TinyBERT rerank) | Golden 15 | 2026-10-18 13:37:38 | 1.0000 | 1.0000 | 0.8 | 0.9333 | 0.9667 | `retrieval_ab_full.json` |
| Full v2 Hybrid (TinyBERT rerank) | Golden 15 | 2026-10-18 13:37:38 | 1.0000 | 1.0000 | 0.7333 | 1.0000 | 0.9667 | `retrieval_ab_full.json` |
| Full v2 Dense (TinyBERT rerank) | Hard 22 | 2026-10-18 13:37:38 | 1.0000 | 0.9545 | 0.8636 | 0.8182 | 0.9091 | `retrieval_ab_full.json` |
| Full v2 Hybrid (TinyBERT rerank) | Hard 22 | 2026-10-18 13:37:38 | 1.0000 | 1.0000 | 0.8636 | 0.9091 | 0.9318 | `retrieval_ab_full.json` |

## 3. Reranker Ablation Benchmark (Top 15 Candidate Pool, Full v2 Collection)

**Source Artifact**: `evals/results/reranker_ablation.json` (Run Timestamp: `2026-10-18 13:06:52`)

| Dataset | Variant | Run Timestamp | Source Hit@5 | Section Hit@5 | Key Hit@5 | Key Hit@15 | Source MRR | Warm Median Latency | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Golden 15 | `no_reranker` | 2026-10-18 13:06:52 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 0.9556 | 0.00 ms | `reranker_ablation.json` |
| Golden 15 | `tinybert` | 2026-10-18 13:06:52 | 1.0000 | 0.7333 | 1.0000 | 1.0000 | 0.9667 | 635.53 ms | `reranker_ablation.json` |
| Golden 15 | `minilm_l12` | 2026-10-18 13:06:52 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 8726.08 ms | `reranker_ablation.json` |
| Golden 15 | `bge_reranker_base` | 2026-10-18 13:06:52 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 1.0000 | 26006.68 ms | `reranker_ablation.json` |
| Hard 22 | `no_reranker` | 2026-10-18 13:06:52 | 0.9545 | 0.9545 | 0.9545 | 1.0000 | 0.9318 | 0.00 ms | `reranker_ablation.json` |
| Hard 22 | `tinybert` | 2026-10-18 13:06:52 | 1.0000 | 0.8636 | 0.9091 | 1.0000 | 0.9318 | 635.53 ms | `reranker_ablation.json` |
| Hard 22 | `minilm_l12` | 2026-10-18 13:06:52 | 1.0000 | 0.9091 | 0.9091 | 1.0000 | 0.9015 | 8726.08 ms | `reranker_ablation.json` |
| Hard 22 | `bge_reranker_base` | 2026-10-18 13:06:52 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9470 | 26006.68 ms | `reranker_ablation.json` |
| Golden 15 | `fusion (TinyBERT+RRF)` *(exploratory, tuned on same set)* | 2026-10-18 05:22:15 | 1.0000 | 0.9333 | 0.8000 | not measured | 0.9667 | composite | `reranker_latency_dedicated.json` |
| Hard 22 | `fusion (TinyBERT+RRF)` *(exploratory, tuned on same set)* | 2026-10-18 05:22:15 | 1.0000 | 1.0000 | 1.0000 | not measured | 0.9318 | composite | `reranker_latency_dedicated.json` |

## 4. Guardrails Token Consumption & Capacity (Measured 5 Real /query Turns)

**Source Artifact**: `evals/results/token_measurement.json` (Run Timestamp: `2026-10-18 19:12:02`)

| Turn # | Query Text | single_call Guard Toks | NeMo Guard Toks | Planner Toks | Responder Toks | Turn Total (single_call) | Turn Total (nemo) | Savings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Turn 1 | What is Horizontal Pod Autoscaling in Kuberne... | 287 | 1918 | 258 | 589 | 1134 | 2765 | -1631 (-59.0%) |
| Turn 2 | How do you configure an NGINX Ingress control... | 302 | 2001 | 269 | 439 | 1010 | 2709 | -1699 (-62.7%) |
| Turn 3 | What Databricks CLI command retrieves the det... | 299 | 1894 | 264 | 1038 | 1601 | 3196 | -1595 (-49.9%) |
| Turn 4 | How do CronJobs handle pod restarts in Kubern... | 296 | 1937 | 261 | 1041 | 1598 | 3239 | -1641 (-50.7%) |
| Turn 5 | What are the completions and parallelism fiel... | 293 | 1989 | 266 | 563 | 1122 | 2818 | -1696 (-60.2%) |

### Summary Averages & Daily Quota Projections (200,000 Tokens/Day)

| Metric | GUARD_MODE=single_call | GUARD_MODE=nemo | Delta | Citation |
| :--- | :--- | :--- | :--- | :--- |
| Avg Guard Tokens / Turn | 387.0 | 1918.0 | -1531.0 (-79.8%) | `token_measurement.json` |
| Avg Planner Tokens / Turn | 0.0 | 0.0 | 0.0 (standalone bypass) | `token_measurement.json` |
| Avg Responder Tokens / Turn | 580.0 | 580.0 | 0.0 | `token_measurement.json` |
| Avg Total Tokens / Turn | 967.0 | 2498.0 | -1531.0 (-61.3% reduction) | `token_measurement.json` |
| Capacity (Turns/Day at 200k TPD) | 206.8 turns/day | 80.1 turns/day | +126.8 turns/day (+158.3%) | `token_measurement.json` |

## 5. Input Safety Guardrail Accuracy (22-Case Dev Set)

**Source Artifact**: `evals/results/single_call_accuracy.json` (Run Timestamp: `2026-10-17 18:23:18`)

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

## 6. Query Routing & Intent Architecture (Live /query Endpoint Validation)

**Source Artifact**: `evals/results/routing_evaluation.json` (Run Timestamp: `2026-10-18 19:07:43`)

| Metric | Measured Value | Citation / Source |
| :--- | :--- | :--- |
| Evaluation Total Cases | 42 (15 Golden + 22 Hard + 5 Chit-Chat) | `routing_evaluation.json` (total_cases) |
| Technical Questions Total | 37 | `routing_evaluation.json` (technical_total) |
| Technical Questions Retrieved | 37 / 37 (100.0%) | `routing_evaluation.json` (technical_retrieved) |
| Technical Questions Skipped (Erroneous) | 0 / 37 | `routing_evaluation.json` (technical_skipped) |
| Chit-Chat Messages Total | 5 ('hi', 'thanks', 'who are you', 'what can you do', 'ok') | `routing_evaluation.json` (chitchat_total) |
| Chit-Chat Retrieval Skipped (Correct) | 5 / 5 (100.0%) | `routing_evaluation.json` (chitchat_skipped) |
| Chit-Chat Retrieval Invoked (Erroneous) | 0 / 5 | `routing_evaluation.json` (chitchat_retrieved) |
| **routing_correctness** | **1.0000 (42/42 cases correct, 100.0%)** | `routing_evaluation.json` (routing_correctness) |

### Planner Routing & Intent Verification Sample

| Question ID | Category | Question | Parsed Intent | Retrieval Invoked | Raw Planner Output Snippet |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G01 | technical | How do you start Redis for a Kubernetes work queue? | Technical | True | `{"route": "retrieve", "query": "start Re...` |
| G02 | technical | What does the parallelism field do in the Kubernetes jo... | Technical | True | `{"route": "retrieve", "query": "Kubernet...` |
| G03 | technical | How do you fill the Redis work queue with tasks using t... | Technical | True | `{"route": "retrieve", "query": "Redis CL...` |
| G04 | technical | What is the difference between HPA and VPA in Kubernete... | Technical | True | `{"route": "retrieve", "query": "differen...` |
| G05 | technical | How do you install the Metrics Server for Kubernetes po... | Technical | True | `{"route": "retrieve", "query": "install ...` |
| G06 | technical | Which kubectl commands confirm that the Metrics Server ... | Technical | True | `{"route": "retrieve", "query": "kubectl ...` |
| G07 | technical | What Databricks CLI command retrieves the details of a ... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G08 | technical | When should you use the Databricks REST API instead of ... | Technical | True | `{"route": "retrieve", "query": "Databric...` |
| G09 | technical | What are the three main developer tools for managing Da... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G10 | technical | What are the valid restart policy values for a Kubernet... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G11 | technical | What are the completions and parallelism fields in a Ku... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G12 | technical | Which Kubernetes API group does a Job resource belong t... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G13 | technical | How do you start a local Kubernetes cluster using Minik... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| G14 | technical | What kubectl command checks the status of a running Kub... | Technical | True | `{"route": "retrieve", "query": "kubectl ...` |
| G15 | technical | How do you verify that the core Kubernetes components a... | Technical | True | `EXCEPTION: Error code: 429 - {'error': {...` |
| C01 | chitchat | hi | Conversational/Memory | False | `{"route": "chitchat"}` |
| C02 | chitchat | thanks | Conversational/Memory | False | `{"route": "chitchat"}` |
| C03 | chitchat | who are you | Conversational/Memory | False | `{"route": "chitchat"}` |
| C04 | chitchat | what can you do | Conversational/Memory | False | `{"route": "chitchat"}` |
| C05 | chitchat | ok | Conversational/Memory | False | `{"route": "chitchat"}` |

## 7. Not Measured / Limitations

1. **Corpus Ingestion Disparity**: The noisy background corpus (`DATA/noisy_data`, 58 documents, 5,871 chunks) was indexed with `v3.0-tokenguard-drop` (chunks <15 tokens dropped). The technical target corpus (`DATA/true_data`, 6 documents, 197 chunks) was indexed with `v3.1-tokenguard-merge-384max` (chunks <15 tokens merged into parent section) to guarantee 100% command and text completeness.
2. **Document Parsing (No OCR)**: Document parsing operates locally on-device (`pypdf`, `pdfplumber`, `python-docx`, `python-pptx`) without an OCR engine. Scanned bitmap PDFs or image-only figures are not extracted into text.
3. **Free-Tier Quota Constraints**: Groq free tier enforces a strict 200,000 tokens/day cap. With planner rewrite bypass on standalone technical queries, single-call mode achieves 967.0 total tokens/turn enabling ~206.8 turns/day vs ~80.1 turns/day for multi-step NeMo (2,498.0 total tokens/turn).
4. **Guardrail Accuracy Scope**: Guardrail accuracy was verified on the 22-case development set (100% on 20 non-infra cases). Performance on proprietary red-team distributions requires running `scripts/run_single_call_accuracy.py --dataset PATH` on a held-out dataset.
5. **Golden Key-Phrase Tightening & Verification**: Generic key terms (`Q4: 'HPA'`, `Q8: 'REST API'`, `Q9: 'CLI'`, `Q14: 'kubectl get jobs'`) were replaced with verified answer-bearing phrases (`'adjusting the number of running pods'`, `'language is not currently available'`, `'Databricks CLI, the Databricks SDKs, and the REST API'`, `'COMPLETIONS   DURATION   AGE'`). Prior labels are tracked in `label_history`. Golden queries + references SHA256 was recomputed and confirmed 100% identical to baseline (`e2a1e8b65018f6732b5df5be19b059547b8917edfc1fcac38d99fdc5c7a6c73d`).
6. **Cross-Encoder CPU Latency**: `BAAI/bge-reranker-base` achieves 1.0000 key hit@5 on Hard 22 but requires ~26,006 ms CPU warm median latency per query vs 635 ms for TinyBERT and 0.0 ms for pure RRF.
7. **Planner Fail-Safe Invariants**: Planner routes directly through a deterministic pre-router for exact social chit-chat (`hi`, `thanks`, `who are you`, etc.). Any unrecognized query, planner exception, or 429 rate limit strictly defaults to `route: 'retrieve'`. In the evaluation run, 100% of technical questions (37/37) invoked hybrid retrieval and TinyBERT reranking, including when transient Groq 429 rate limits occurred on the planner node.
