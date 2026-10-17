"""
Phase 2 — RAGAS + Tool Correctness metrics.
Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.
All LLM-based metrics run in batches of 5 with 30s cooldowns between sub-batches
and 60s cooldowns between experiments — calibrated for Groq's 6,000 TPM on_demand tier.
Contexts are truncated to 300 chars (2 chunks max) so no single request exceeds the limit.
"""


import os
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI
from app.config import settings


from langchain_openai import ChatOpenAI
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "llama-3.1-8b-instant"
COOLDOWN_STANDARD = 0
COOLDOWN_MINI = 0
GENERAL_BATCH_SIZE = 3  # 3 samples per mini-batch (~1,200 tok/batch well under 6,000 TPM limit)
CONTEXT_TRUNCATE = 300  # chars per context chunk — reduces single request from ~7,700 to ~400 tokens
CONTEXT_LIMIT = 2       # number of context chunks passed to RAGAS per sample


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    judge_model = os.getenv("JUDGE_MODEL", settings.GROQ_MODEL)
    chat_model = ChatOpenAI(
        model=judge_model,
        openai_api_key=api_key,
        openai_api_base=GROQ_BASE_URL,
        temperature=0,
        max_retries=0
    )
    llm = LangchainLLMWrapper(chat_model)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks
    so a single RAGAS LLM call stays well under the 6,000 TPM ceiling.
    (Live contexts from Qdrant are ~1,500 chars each — without truncation a single
    Faithfulness request exceeds 7,000 tokens which hard-fails on the on_demand tier.)
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    formatted_scores = []
    for r in scores:
        if hasattr(r, "value"):
            val = float(r.value)
        elif hasattr(r, "__float__"):
            val = float(r)
        else:
            try:
                val = float(r)
            except Exception:
                val = 0.0
        formatted_scores.append(round(val, 3))
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: val}
        for s, val in zip(samples, formatted_scores)
    ])


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Evaluates samples via single_turn_ascore with mini-batch pacing.
    Keeps each burst under 6,000 TPM on Groq's on_demand tier.
    """
    all_scores = []
    for idx, inp in enumerate(inputs):
        if idx > 0 and idx % GENERAL_BATCH_SIZE == 0:
            await _cooldown(COOLDOWN_MINI, f"{label} sample {idx}", status_cb)
        sample_obj = SingleTurnSample(**inp)
        try:
            score = await metric.single_turn_ascore(sample_obj)
        except Exception as e:
            logfire.warning(f"Error scoring sample {idx} for {label}: {e}")
            score = 0.0
        all_scores.append(score)
    return all_scores

async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness (split into batches) ────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness batch 1/2...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results


def compute_retrieval_metrics(samples: list) -> dict:
    """
    Computes deterministic retrieval-only metrics (zero LLM calls):
      - source_hit@5: proportion of queries where expected_source is in top-5 retrieved chunks (required)
      - source_mrr: Mean Reciprocal Rank of first occurrence of expected_source
      - section_hit@5: proportion of queries where expected_source AND expected_section match in top-5
      - section_mrr: Mean Reciprocal Rank of first occurrence where both match
    """
    src_hits_at_5 = 0
    src_rr_total = 0.0
    sec_hits_at_5 = 0
    sec_rr_total = 0.0
    valid_count = 0

    per_sample_results = []

    for s in samples:
        expected_src = s.get("expected_source")
        expected_sec = s.get("expected_section")
        if not expected_src:
            continue
        valid_count += 1

        contexts = s.get("actual_contexts", [])
        src_ranks = []
        sec_ranks = []

        for idx, ctx in enumerate(contexts):
            rank = idx + 1
            src_str = ""
            sec_str = ""

            if isinstance(ctx, dict):
                src_str = ctx.get("source", "")
                sec_str = ctx.get("section", "")
            elif isinstance(ctx, str):
                # Standard format: "[Source: <source> | Section: <section>]\n<content>"
                if "[Source:" in ctx:
                    parts = ctx.split("[Source:", 1)[1]
                    if "|" in parts:
                        src_part, rest = parts.split("|", 1)
                        src_str = src_part.strip()
                        if "Section:" in rest:
                            sec_str = rest.split("Section:", 1)[1].split("]", 1)[0].strip()
                    elif "]" in parts:
                        src_str = parts.split("]", 1)[0].strip()
                else:
                    src_str = ctx

            # 1. Source match (required)
            matches_source = expected_src.lower() in src_str.lower()
            if matches_source:
                src_ranks.append(rank)

            # 2. Section match (reported separately)
            if matches_source and expected_sec:
                # Direct substring match between expected_section and retrieved section
                if (expected_sec.lower() in sec_str.lower() or
                    sec_str.lower() in expected_sec.lower() or
                    expected_sec.lower() in (ctx if isinstance(ctx, str) else ctx.get("content", "")).lower()):
                    sec_ranks.append(rank)

        # Source Hit & MRR
        src_first_rank = src_ranks[0] if src_ranks else None
        src_hit = 1.0 if (src_first_rank and src_first_rank <= 5) else 0.0
        src_rr = 1.0 / src_first_rank if src_first_rank else 0.0
        src_hits_at_5 += src_hit
        src_rr_total += src_rr

        # Section Hit & MRR
        sec_first_rank = sec_ranks[0] if sec_ranks else None
        sec_hit = 1.0 if (sec_first_rank and sec_first_rank <= 5) else 0.0
        sec_rr = 1.0 / sec_first_rank if sec_first_rank else 0.0
        sec_hits_at_5 += sec_hit
        sec_rr_total += sec_rr

        per_sample_results.append({
            "question": s.get("question", "")[:65],
            "expected_source": expected_src,
            "expected_section": expected_sec,
            "source_rank": src_first_rank,
            "source_hit@5": src_hit,
            "source_rr": round(src_rr, 3),
            "section_rank": sec_first_rank,
            "section_hit@5": sec_hit,
            "section_rr": round(sec_rr, 3),
        })

    total = valid_count if valid_count > 0 else 1
    return {
        "hit@5": round(src_hits_at_5 / total, 3),
        "mrr": round(src_rr_total / total, 3),
        "source_hit@5": round(src_hits_at_5 / total, 3),
        "source_mrr": round(src_rr_total / total, 3),
        "section_hit@5": round(sec_hits_at_5 / total, 3),
        "section_mrr": round(sec_rr_total / total, 3),
        "details": per_sample_results
    }


