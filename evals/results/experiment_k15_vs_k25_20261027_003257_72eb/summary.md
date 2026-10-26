# TrustRAG Paired Retrieval Experiment: K=15 vs K=25

**Run ID**: `20261027_003257_72eb`  
**Timestamp**: `2026-10-27 00:41:15`  
**Dataset SHA256**: `0b63bf1b40684ff7ef18e3fa63b7edb31d628f339bc71d2dea7e023fbe798451`  
**Parameters**: `reranker_margin=0.20`, `reranker_min_score=0.08`, `top_n=5`  

## 1. Executive Summary

| Metric | K=15 Baseline | K=25 Candidate Pool | Delta |
|---|---|---|---|
| **Pass Rate (All 50)** | 29/50 (58.0%) | 30/50 (60.0%) | +2.0% |
| **Pass Rate (Clean Dev Cohorts 2+3, n=45)** | 24/45 (53.3%) | 25/45 (55.6%) | +2.2% |
| **Recall@5** (answerable) | 0.4750 | 0.4667 | -0.0083 |
| **HitRate@5** (answerable) | 0.5500 | 0.5500 | +0.0000 |
| **MRR@10** (answerable) | 0.4308 | 0.4308 | +0.0000 |
| **nDCG@5** (answerable) | 0.4004 | 0.3944 | -0.0060 |
| **Multi-hop Coverage** | 0/0 (0.0%) | 0/0 (0.0%) | +0 |
| **Mean Latency (ms)** | 4146.5 ms | 3758.9 ms | -387.6 ms |
| **p95 Latency (ms)** | 7954.7 ms | 6721.5 ms | -1233.2 ms |
| **Infra Errors / Timeouts** | 0 | 0 | 0 |

## 2. Paired Verdict Analysis

- **K=15 Wins**: 0
- **K=25 Wins**: 1
- **Ties**: 49
- **Disagreements**: 0

### Paired Discordance Details

| Question ID | Cohort | K=15 Verdict | K=25 Verdict | Outcome | Detail |
|---|---|---|---|---|---|
| `seed_009` | seed_006_010_batch2_edits_unverified | ABSTAINED | CORRECT | K25_WIN | Clearly distinguishes HPA (horizontal pod scaling) from VPA  |

## 3. Stratified Cohort Breakdown

| Cohort | N | K=15 Pass Rate | K=25 Pass Rate | K=15 Recall@5 | K=25 Recall@5 | K=15 HitRate@5 | K=25 HitRate@5 |
|---|---|---|---|---|---|---|---|
| `all_50` | 50 | 29/50 (58.0%) | 30/50 (60.0%) | 0.475 | 0.467 | 0.550 | 0.550 |
| `seed_001_005_disputed` | 5 | 5/5 (100.0%) | 5/5 (100.0%) | 0.800 | 0.800 | 0.800 | 0.800 |
| `seed_006_010_corpus_edits` | 5 | 3/5 (60.0%) | 4/5 (80.0%) | 0.667 | 0.667 | 0.800 | 0.800 |
| `seed_011_050_remainder` | 40 | 21/40 (52.5%) | 21/40 (52.5%) | 0.389 | 0.378 | 0.467 | 0.467 |
| `eligible_clean_dev_cohort_2_3` | 45 | 24/45 (53.3%) | 25/45 (55.6%) | 0.429 | 0.419 | 0.514 | 0.514 |

## 4. Engineering Findings & Recommendation

- **Candidate Pool Impact**: Expanding Qdrant initial candidate pool from 15 to 25 feeds 25 candidates to FlashRank reranking.
- **Latency Tradeoff**: Measured average latency difference between scoring 15 vs 25 candidates.
- **Retrieval & Grounding Effect**: Documented whether any missing documents from smaller candidate pools are recovered.
