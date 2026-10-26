# Phase 2 vs Baseline Comparison Report
> [!WARNING]
> **STATUS: INVALID - JUDGE QUOTA FAILURES**
> This evaluation report is marked invalid due to free-tier Groq daily limit exhaustion (200,000 Tokens Per Day) during offline LLM judge evaluation (>80% failure rate). RESULTS.md does not cite this file.
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
| **Source Hit@15 (Pre-rerank)**| 1.000 | 1.000 | 0.000 | ✅ 100% Pre-rerank Source Hit (formerly labeled Recall@15) |
| **Corpus Point Count** | 276 (unfiltered) | 6,070 (199 True + 5,871 Noisy) | +5,794 | ✅ Distraction Stress Test |

## 3. LLM-Judged Metrics (Omitted due to >10% Quota Failures & Pipeline Artifacts)
- `faithfulness`: 86.7% failure rate (daily TPD limit reached)
- `answer_relevancy`: 80.0% failure rate (daily TPD limit reached)
- `context_precision`: 93.3% failure rate (daily TPD limit reached)
- `context_recall`: 93.3% failure rate (daily TPD limit reached)
- `answer_correctness`: 86.7% failure rate (daily TPD limit reached)

### Methodological Invalidation Notes:
1. **API Quota Exceptions Converted to Zero**: In `evals/metrics.py:116-118`, unhandled HTTP 429 rate limit exceptions were caught and assigned a score of `0.0`. Reported baseline averages (e.g., Faithfulness 0.133, Context Precision 0.067 in `baseline.json`) represent rate-limit dropouts rather than true model performance.
2. **Context & Response Truncation (300 Chars)**: In `evals/pipeline.py:17` (`RESPONSE_TRUNCATE = 300`) and `evals/metrics.py:36-37` (`CONTEXT_TRUNCATE = 300`, `CONTEXT_LIMIT = 2`), retrieved contexts and generated answers were truncated to 300 characters (~80% of context removed). The judge LLM was asked to evaluate factual faithfulness against truncated snippets, generating severe false-negative hallucination penalties.
3. *Conclusion:* Deterministic Information Retrieval metrics (chunk-level Recall, Precision, Hit Rate, MRR, nDCG) and guardrail security benchmarks remain the authoritative, reproducible evaluation baseline.
