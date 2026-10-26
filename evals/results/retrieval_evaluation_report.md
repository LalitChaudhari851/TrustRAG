# TrustRAG Retrieval Evaluation Report

> [!WARNING]
> **UNVERIFIED SEED DATASET NOTICE**
> NOTICE: ALL EVALUATED RECORDS ARE UNVERIFIED SYNTHETIC SEED CANDIDATES. RESULTS REFLECT INITIAL SEED PERFORMANCE AND MUST NOT BE CITED AS A VERIFIED BENCHMARK.

## 1. Evaluation & Environment Manifest

| Manifest Property | Configuration |
| :--- | :--- |
| **Target Qdrant Collection** | `enterprise_rag_v2` |
| **Total Indexed Points** | `6068` |
| **True Knowledge Points** | `197` |
| **Dense Embedding Model** | `sentence-transformers/all-mpnet-base-v2 (768-dim, Cosine)` |
| **Sparse Embedding Model** | `FastEmbed BM25 (Modifier.IDF)` |
| **Reranker Engine** | `ms-marco-TinyBERT-L-2-v2 (FlashRank locally cached)` |
| **Reranker Floor / Margin** | `floor=0.08, margin=0.2` |
| **Dataset Version & Split** | `evals/data/retrieval_golden_seed_v1.json` (Split: `dev`, N=30) |
| **Execution Timestamp** | `2026-10-25T19:02:04.798801+00:00` |

## 2. Results: Full Corpus (Production Realism with 5,871 Distractor Points)

### Relevance Macro-Averages (Answerable Queries Only)

| Pipeline | Recall@5 | Recall@10 | Precision@5 (Fixed) | Precision@5 (Actual) | HitRate@5 | MRR@10 | Graded nDCG@10 | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 0.660 [0.486, 0.833] | 0.660 | 0.167 | 0.167 | 0.708 | 0.494 | 0.492 [0.350, 0.634] | 1215.81ms |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 0.549 [0.361, 0.743] | 0.549 | 0.133 | 0.258 | 0.625 | 0.468 | 0.441 [0.289, 0.619] | 2367.15ms |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 0.604 [0.417, 0.792] | 0.604 | 0.150 | 0.150 | 0.667 | 0.485 | 0.464 [0.308, 0.612] | 400.2ms |

*Note: Bracketed ranges denote 95% bootstrap confidence intervals (1,000 resamples). Sample size N=24 answerable queries.*

### Multi-Hop Conjunction Metrics

| Pipeline | Evidence Coverage@5 | All-Required Coverage@5 | Multi-Hop Success@5 | Evidence Coverage@10 | All-Required Coverage@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 0.583 | 0.500 | 0.500 | 0.583 | 0.500 |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 0.417 | 0.167 | 0.167 | 0.417 | 0.167 |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 0.500 | 0.333 | 0.333 | 0.500 | 0.333 |

### Distractor Interference

| Pipeline | Distractors Before First Relevant@5 | Distractor Ratio@5 | Distractor Ratio@10 |
| :--- | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 2.17 | 0.833 | 0.833 |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 2.08 | 0.742 | 0.742 |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 2.38 | 0.850 | 0.850 |

### Retrieval Abstention on Unanswerable Queries

| Pipeline | Unanswerable Queries | Correct Retrieval Abstentions | Retrieval Abstention Rate |
| :--- | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 6 | 0 | 0.0% |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 6 | 3 | 50.0% |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 6 | 0 | 0.0% |

---

## 2. Results: True-Only Corpus (Isolated 197 True Knowledge Units)

### Relevance Macro-Averages (Answerable Queries Only)

| Pipeline | Recall@5 | Recall@10 | Precision@5 (Fixed) | Precision@5 (Actual) | HitRate@5 | MRR@10 | Graded nDCG@10 | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 0.660 [0.486, 0.833] | 0.660 | 0.167 | 0.167 | 0.708 | 0.510 | 0.503 [0.355, 0.650] | 391.76ms |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 0.549 [0.361, 0.743] | 0.549 | 0.133 | 0.258 | 0.625 | 0.468 | 0.441 [0.289, 0.619] | 1727.03ms |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 0.604 [0.417, 0.792] | 0.604 | 0.150 | 0.150 | 0.667 | 0.487 | 0.466 [0.310, 0.614] | 468.73ms |

*Note: Bracketed ranges denote 95% bootstrap confidence intervals (1,000 resamples). Sample size N=24 answerable queries.*

### Multi-Hop Conjunction Metrics

| Pipeline | Evidence Coverage@5 | All-Required Coverage@5 | Multi-Hop Success@5 | Evidence Coverage@10 | All-Required Coverage@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 0.583 | 0.500 | 0.500 | 0.583 | 0.500 |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 0.417 | 0.167 | 0.167 | 0.417 | 0.167 |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 0.500 | 0.333 | 0.333 | 0.500 | 0.333 |

### Distractor Interference

| Pipeline | Distractors Before First Relevant@5 | Distractor Ratio@5 | Distractor Ratio@10 |
| :--- | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 2.21 | 0.833 | 0.833 |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 2.08 | 0.742 | 0.742 |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 2.33 | 0.850 | 0.850 |

### Retrieval Abstention on Unanswerable Queries

| Pipeline | Unanswerable Queries | Correct Retrieval Abstentions | Retrieval Abstention Rate |
| :--- | :---: | :---: | :---: |
| **Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)** | 6 | 0 | 0.0% |
| **Hybrid RRF + FlashRank TinyBERT (Production Path)** | 6 | 3 | 50.0% |
| **Dense-Only (Single MPNet Vector, Raw Top-K)** | 6 | 0 | 0.0% |

---

## 3. Methodological Notes

1. **Dual Precision Conventions**: Precision@5 (Fixed) divides hits by 5, penalizing pipelines when fewer than 5 items are returned. Precision@5 (Actual) divides hits by the candidate count actually returned by FlashRank after score filtering.
2. **Unanswerable Isolation**: Unanswerable queries ($N=10$) are isolated from relevance macro-averages to prevent $0/0$ distortion, and are evaluated under Abstention Rate.
3. **Multi-Hop Conjunction**: Multi-hop queries evaluate whether *all* required evidence chunks from at least one sufficient set are retrieved in top-K, not merely whether any random chunk was hit.