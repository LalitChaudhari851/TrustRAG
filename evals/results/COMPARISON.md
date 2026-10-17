# Phase 2 vs Baseline Comparison Report

## 1. Executive Summary & Verification Disclosures
- **Judge Model Disclosures:** The baseline specification referenced `llama-3.1-8b-instant`. However, this model returned HTTP 404 (de-listed from the Groq organization catalog). The live judge model utilized is `qwen/qwen3.8-27b` (via Portkey/Groq) at `temperature=0.0`. Embeddings used for RAGAS evaluation are local `sentence-transformers/all-mpnet-base-v2` on CPU.
- **Quota & Failure Rate Constraints:** Free-tier Groq daily limit (200,000 Tokens Per Day) experienced exhaustion during live RAGAS testing. In accordance with benchmark rules, **only metrics with <10% failures are reported as conclusive**.
- **Guard-Outage Fallthrough Disclosure:** The guardrail rate-limit fallthrough (`is_rail_infra -> fall through to agent`) exists **strictly within the offline evaluation script (`scripts/run_ragas_after_phase2.py`)** to avoid test termination. In production application code (`app/main.py`), guardrail activation or outage **immediately terminates the request without fallthrough**.

## 2. Deterministic Retrieval Comparison (<10% Failures)

| Metric | Legacy Baseline (`enterprise_rag`) | Phase 2 Hybrid (`enterprise_rag_v2`) | Delta | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Source Hit@5** | 1.000 | 1.000 | 0.000 | ✅ 100% Retained |
| **Source MRR** | 0.906 | 0.933 | +0.027 | ✅ Improved |
| **Section Hit@5** | n/a (General fallback) | 0.800 (12/15) | +0.800 | ✅ Structured Section Resolution |
| **Section MRR** | n/a (General fallback) | 0.722 | +0.722 | ✅ High-Precision Section MRR |
| **Recall@15 (Pre-rerank)**| 1.000 | 1.000 | 0.000 | ✅ 100% Pre-rerank Recall |
| **Corpus Point Count** | 276 (unfiltered) | 6,070 (199 True + 5,871 Noisy) | +5,794 | ✅ Distraction Stress Test |

## 3. LLM-Judged Metrics (Omitted due to >10% Quota Failures)
- `faithfulness`: 86.7% failure rate (daily TPD limit reached)
- `answer_relevancy`: 80.0% failure rate (daily TPD limit reached)
- `context_precision`: 93.3% failure rate (daily TPD limit reached)
- `context_recall`: 93.3% failure rate (daily TPD limit reached)
- `answer_correctness`: 86.7% failure rate (daily TPD limit reached)
- *Note:* Multi-day phased evaluation or Tier-1 Groq quota is required to compute statistically sound RAGAS scores.
