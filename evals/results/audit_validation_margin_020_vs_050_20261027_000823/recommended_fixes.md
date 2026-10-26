# Recommended Fixes: Evaluation Engine, Retrieval & Annotation Workflow

Based on the automated audit of the 50-question paired validation, the following concrete fixes are recommended:

---

## 1. Evaluator Engine Fixes (`scripts/run_batch_eval.py`)

### Fix 1.1: Contextualize `is_abstention_answer()` String Matching
- **Problem:** Phrases like `"does not provide"` or `"does not mention"` currently trigger an immediate `ABSTAINED` verdict even when the model provides a 150-word detailed factual answer and only includes that phrase in a closing disclaimer (e.g., `seed_021` and `seed_040`).
- **Fix:** Restrict refusal phrase detection to:
  1. Responses shorter than 40 words, OR
  2. Responses where the refusal phrase appears within the first 60 characters (e.g., `"I cannot answer this question because..."`).

### Fix 1.2: Add Deterministic Unit Rules for Core Entities
- **Problem:** Generic word-token overlap produced a false negative on `seed_014` (kubelet) and a false regression on `seed_030` (Databricks CLI create).
- **Fix:** Add deterministic match rules in `DeterministicEvaluator` for:
  - `seed_014`: Accept `kubelet` + (`node agent` or `pod lifecycle` or `ensures containers`).
  - `seed_030`: Accept `databricks jobs create` + `--json`.

### Fix 1.3: Align Safety Guardrail Refusals on Unanswerable Queries
- **Problem:** In `seed_047`, safety guardrails intercepted an unanswerable query, resulting in an `ABSTAINED` verdict instead of `CORRECT`.
- **Fix:** If `rec.query_type == "unanswerable"` and `raw_status in ("blocked_by_guardrails", "insufficient_context")`, assign `verdict = "CORRECT"`.

---

## 2. Retrieval Configuration Fixes

### Fix 2.1: Promote `reranker_margin=0.50` to Primary Staging Candidate
- Margin 0.50 recovered 3 critical questions (`seed_008`, `seed_009`, `seed_040`) with zero real regressions, reduced tail latency by 1.18 seconds, and increased context by only +0.28 chunks.
- Keep `floor=0.08` active as the safety floor against irrelevant candidate pools.

### Fix 2.2: Expand Upstream Hybrid Candidate Pool ($K=15 	o 25$)
- Audit showed that 15 answerable abstentions failed not because of reranker pruning, but because the target chunk was not in the top 15 hybrid search results.
- A controlled candidate pool sweep ($K=20, 25, 30$) is the single highest-leverage next step for overall system recall.

---

## 3. Human Annotation Workflow Fixes

1. **Reconcile `seed_001`–`seed_005`:** Schedule human review to confirm the Redis work queue question mapping and record a fresh human verification event.
2. **Review Queue Clearance:** Process the 23 items isolated in `question_level_audit.csv` needing human verification.
