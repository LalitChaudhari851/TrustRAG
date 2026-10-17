import os
import tempfile
import time
from typing import Union, List, Dict, Any
import logfire
from flashrank import Ranker, RerankRequest
from app.config import settings

_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily using a cross-platform cache directory.
    """
    global _ranker
    if _ranker is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
        os.makedirs(cache_dir, exist_ok=True)
        logfire.info(f"🧠 Initializing FlashRank Model (TinyBERT) locally (cache_dir={cache_dir})...")
        try:
            _ranker = Ranker(cache_dir=cache_dir)
        except Exception as e:
            logfire.warning(f"Failed to initialize Ranker with cache_dir={cache_dir}: {e}. Retrying default.")
            _ranker = Ranker()
    return _ranker


def rerank_documents(
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    top_n: int = 5,
    min_score: float | None = None,
    relative_margin: float | None = None
) -> List[Any]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    Applies calibrated dual-threshold filtering:
      1. Absolute floor (settings.RERANKER_MIN_SCORE): rejects when top_score < floor
      2. Relative margin (settings.RERANKER_RELATIVE_MARGIN): drops chunks scoring below
         top_score * (1 - margin) to eliminate low-relevance clutter.
    
    Accepts either:
      - list[str]: returns list[str]
      - list[dict]: returns list[dict] with enriched 'rerank_score'
    """
    if not documents:
        return []

    floor = settings.RERANKER_MIN_SCORE if min_score is None else min_score
    margin = settings.RERANKER_RELATIVE_MARGIN if relative_margin is None else relative_margin
    start_time = time.perf_counter()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank (floor={floor}, margin={margin})...")

    is_dict_input = isinstance(documents[0], dict)

    try:
        ranker = _get_ranker()

        passages = []
        for i, doc in enumerate(documents):
            text = doc["content"] if is_dict_input else str(doc)
            passages.append({"id": i, "text": text})

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        rerank_duration_ms = (time.perf_counter() - start_time) * 1000.0

        if not results:
            logfire.info("Reranker returned zero candidates.")
            return []

        top_score = float(results[0].get("score", 0.0))
        if top_score < floor:
            logfire.info(f"Top rerank score {top_score:.4f} < floor {floor:.4f} — refusing context.")
            return []

        relative_cutoff = max(floor, top_score * (1.0 - margin))

        reranked_outputs = []
        dropped_count = 0

        for res in results:
            score = float(res.get("score", 0.0))
            if score < relative_cutoff:
                dropped_count += 1
                continue

            orig_idx = int(res["id"])
            if is_dict_input:
                doc_copy = dict(documents[orig_idx])
                doc_copy["rerank_score"] = score
                doc_copy["rerank_duration_ms"] = rerank_duration_ms
                reranked_outputs.append(doc_copy)
            else:
                reranked_outputs.append(res["text"])

            if len(reranked_outputs) >= top_n:
                break

        logfire.info(
            f"✅ [Reranker] Done in {rerank_duration_ms:.1f}ms. "
            f"Top score: {top_score:.4f}, cutoff: {relative_cutoff:.4f}, kept: {len(reranked_outputs)}, dropped: {dropped_count}"
        )
        return reranked_outputs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original order
        return documents[:top_n]
