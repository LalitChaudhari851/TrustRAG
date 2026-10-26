# TrustRAG Batch Evaluation Report

**Run ID:** `batch_20261026_232510_240f`  
**Timestamp:** `2026-10-26T23:25:54.590713`  
**Endpoint:** `http://127.0.0.1:8000/query`  
**Selected IDs:** `seed_001`, `seed_002`, `seed_003`, `seed_004`, `seed_005`, `seed_006`, `seed_007`, `seed_008`, `seed_009`, `seed_010`  
**Evaluator:** `deterministic_v1`  

---

## 1. Executive Performance Summary

| Metric | Count / Score | Denominator / Notes |
| :--- | :--- | :--- |
| **Total Evaluated** | **10** | Selected dataset batch |
| **Correct (Passed)** | **8** | Full factual alignment |
| **Partially Correct** | **0** | Partial key information present |
| **Incorrect** | **0** | Factual error or contradiction |
| **Abstained** | **2** | Grounding refusal ("Not Found in Documentation") |
| **Infrastructure Errors** | **0** | Network / Timeout / 5xx failures |
| **Not Evaluable** | **0** | Incomplete response or missing reference |
| **Answer Pass Rate** | **80.0%** | `8/10 evaluable queries` |

> [!NOTE]
> Infrastructure failures (0) are isolated and excluded from the answer accuracy denominator to prevent hiding systemic service outages inside model answer failures.

---

## 2. Macro Retrieval Metrics (Top-K)

*Evaluated on `10` eligible answerable queries with annotated relevance grades.*

| Metric | Score | Sample Size / Notes |
| :--- | :--- | :--- |
| **Recall@1** | **0.6000** | Primary top-1 rank hit rate |
| **Recall@5** | **0.7333** | Default production context window |
| **Recall@10** | **0.7333** | Reranker candidate pool depth |
| **HitRate@1** | **0.6000** | Any relevant chunk at Rank 1 |
| **HitRate@5** | **0.8000** | Any relevant chunk in Top 5 |
| **HitRate@10** | **0.8000** | Any relevant chunk in Top 10 |
| **MRR@10** | **0.6533** | Mean Reciprocal Rank |
| **nDCG@5** | **0.6440** | Normalized Discounted Cumulative Gain |
| **nDCG@10** | **0.6440** | Full rerank list ranking quality |

---

## 3. Per-Question Results Table

| ID | Query | Verdict | Latency | Relevant Retrieved? | Abstention Status | Diagnosis / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `seed_001` | How do you start Redis for a Kubernetes work queue? | **CORRECT** | 4340ms | ✅ Yes | `none` | Cites both redis-pod.yaml and redis-service.yaml manifests with kubectl apply. |
| `seed_002` | Which CLI commands push tasks into the Redis queue named job2? | **CORRECT** | 3714ms | ✅ Yes | `none` | Correctly cites rpush job2 command syntax to enqueue tasks. |
| `seed_003` | What parallelism value is specified in the job-wq-2 manifest? | **CORRECT** | 3883ms | ✅ Yes | `none` | Correctly states parallelism value of 2. |
| `seed_004` | What restart policies are permitted for Kubernetes Job pods? | **CORRECT** | 4964ms | ✅ Yes | `none` | Correctly identifies OnFailure and Never policies. |
| `seed_005` | Which API group and version defines the Kubernetes Job resource? | **CORRECT** | 7162ms | ❌ No | `none` | Correctly states the batch/v1 API group and version. |
| `seed_006` | What does the backoffLimit field specify in a Kubernetes Job configuration? | **CORRECT** | 7819ms | ✅ Yes | `none` | Accurately explains backoffLimit specifies the number of retries before Job failure. |
| `seed_007` | What does activeDeadlineSeconds do when applied to a Job? | **CORRECT** | 2481ms | ✅ Yes | `none` | Explains active execution deadline terminating/killing the Job once exceeded. |
| `seed_008` | How can you configure a Kubernetes CronJob to automatically clean up completed jobs after 1 hour to minimize costs? | **ABSTAINED** | 2452ms | ❌ No | `refused_insufficient_context` | RELEVANT_NOT_RETRIEVED |
| `seed_009` | What is the core difference between HPA and VPA in Kubernetes? | **ABSTAINED** | 3694ms | ✅ Yes | `refused_insufficient_context` | RELEVANT_DROPPED_BY_RERANKER |
| `seed_010` | How do you install the Metrics Server in Kubernetes using the official components manifest? | **CORRECT** | 3406ms | ✅ Yes | `none` | Provides the official kubectl apply command with components.yaml manifest. |

---

## 4. Abstention & Root-Cause Diagnosis

This runner distinguishes the following explicit abstention failure modes:
1. **RELEVANT_NOT_RETRIEVED**: The annotated candidate chunk was not returned in Qdrant hybrid retrieval top-K.
2. **RELEVANT_DROPPED_BY_RERANKER**: Candidate chunk retrieved in initial search but dropped below the FlashRank score floor.
3. **CONTEXT_PRESENT_MODEL_ABSTAINED**: Relevant chunk was present in the prompt context, but the responder refused to answer.
4. **GUARDRAIL_BLOCKED**: Safety guardrails intercepted the query before retrieval.
5. **INFRASTRUCTURE_FAILURE**: HTTP timeout, network disconnect, or rate limit 429.

### Question-by-Question Diagnoses:
- **`seed_008`**: Verdict `ABSTAINED` | Diagnosis: `RELEVANT_NOT_RETRIEVED`  
  *Details*: None of candidate chunks ['8f7910c9-d6a1-5234-950f-ed26139ea9a3'] were retrieved in top-K.  
- **`seed_009`**: Verdict `ABSTAINED` | Diagnosis: `RELEVANT_DROPPED_BY_RERANKER`  
  *Details*: Relevant chunk was retrieved in vector search but dropped before final LLM context.  

---

## 5. Audit & Runtime Configuration

- **Dataset Path:** `evals/data/retrieval_golden_seed_v1.json`
- **Server URL:** `http://127.0.0.1:8000`
- **Auth Mode:** `none`
- **Server Groq Model:** `qwen/qwen3.8-27b`
- **Server Guard Mode:** `single_call`
- **Active Guard Model:** `openai/gpt-oss-20b`

> [!WARNING]
> **Sample Size Notice**: This evaluation was executed on a development sample of N=10 queries. While highly informative for local regression testing, production deployment sign-off requires running the full 50-query golden benchmark with verified human ground truth.
