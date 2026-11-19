import os
import time
import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query

LOCAL_QDRANT_PATH = "data/local_qdrant"


def get_qdrant_client() -> QdrantClient:
    """
    Initializes Qdrant client with automatic failover to local embedded storage
    if the remote cluster is unreachable or unconfigured.
    """
    url = getattr(settings, "QDRANT_URL", None)
    api_key = getattr(settings, "QDRANT_API_KEY", None)

    if not url or url.lower() in ("local", ":memory:", "none") or os.path.exists(url):
        target = url if (url and os.path.exists(url)) else LOCAL_QDRANT_PATH
        logfire.info(f"Using local embedded Qdrant storage at '{target}'")
        return QdrantClient(path=target)

    try:
        remote_client = QdrantClient(url=url, api_key=api_key, timeout=3.0)
        remote_client.get_collections()
        logfire.info(f"Connected to remote Qdrant cluster at {url}")
        return remote_client
    except Exception as e:
        logfire.warning(
            f"⚠️ Remote Qdrant cluster '{url}' unreachable ({e}). "
            f"Failing over to local embedded storage at '{LOCAL_QDRANT_PATH}'."
        )
        return QdrantClient(path=LOCAL_QDRANT_PATH)


# Initialize Qdrant Client (with automatic resilient fallback)
client = get_qdrant_client()


def compute_python_rrf(dense_ranks: list[str], sparse_ranks: list[str], k: int = 60) -> list[tuple[str, float]]:
    """
    Plain-Python Reciprocal Rank Fusion (RRF) implementation.
    Kept strictly for unit testing and offline verification.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_enterprise_knowledge(
    query: str,
    limit: int = 15,
    hybrid: bool | None = None,
    query_filter: models.Filter | None = None
) -> list[dict]:
    """
    Performs high-precision vector search in the enterprise knowledge base.
    Uses Qdrant native hybrid (Prefetch + FusionQuery(RRF)) when hybrid is enabled,
    or named dense vector search when hybrid is disabled.
    
    Returns a list of dicts containing chunk content and rich document metadata.
    """
    use_hybrid = settings.HYBRID_SEARCH if hybrid is None else hybrid
    collection = settings.QDRANT_COLLECTION

    t_embed_start = time.perf_counter()
    dense_vector = embed_query(query)
    sparse_vector = None
    if use_hybrid:
        try:
            sparse_vector = embed_sparse_query(query)
        except Exception as e:
            logfire.warning(f"Failed to generate sparse query embedding: {e}. Falling back to dense-only.")
            use_hybrid = False
    embed_latency_ms = (time.perf_counter() - t_embed_start) * 1000.0

    t_qdrant_start = time.perf_counter()
    try:
        if use_hybrid and sparse_vector is not None:
            logfire.info(f"🔎 Executing Qdrant Native Hybrid Search (Dense + Sparse RRF) on '{collection}' (filter={bool(query_filter)})")
            response = client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=limit, filter=query_filter),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=limit, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            logfire.info(f"🔎 Executing Qdrant Dense-Only Search on '{collection}' (filter={bool(query_filter)})")
            try:
                # Named dense vector in enterprise_rag_v2
                response = client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    using="dense",
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
            except Exception as named_err:
                # Fallback for collections with unnamed single vector (e.g. legacy enterprise_rag)
                logfire.info(f"Named 'dense' vector query failed ({named_err}), falling back to unnamed vector.")
                response = client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )

        qdrant_latency_ms = (time.perf_counter() - t_qdrant_start) * 1000.0

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "id": str(res.id),
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "source_type": payload.get("source_type", ""),
                "page": payload.get("page"),
                "total_pages": payload.get("total_pages"),
                "slide": payload.get("slide"),
                "slide_title": payload.get("slide_title"),
                "section": payload.get("section"),
                "chunk_index": payload.get("chunk_index"),
                "score": res.score,
                "latency_stage": {
                    "embed_ms": embed_latency_ms,
                    "qdrant_ms": qdrant_latency_ms,
                }
            })

        logfire.info(
            f"✅ Qdrant search returned {len(results)} points "
            f"(embed: {embed_latency_ms:.1f}ms, qdrant: {qdrant_latency_ms:.1f}ms)"
        )
        return results

    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed on {collection}: {e}")
        # If remote search failed, attempt one fallback retry on local storage
        try:
            logfire.warning(f"Retrying query against local embedded storage '{LOCAL_QDRANT_PATH}'...")
            local_client = QdrantClient(path=LOCAL_QDRANT_PATH)
            if use_hybrid and sparse_vector is not None:
                response = local_client.query_points(
                    collection_name=collection,
                    prefetch=[
                        models.Prefetch(query=dense_vector, using="dense", limit=limit, filter=query_filter),
                        models.Prefetch(query=sparse_vector, using="sparse", limit=limit, filter=query_filter),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                )
            else:
                response = local_client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    using="dense",
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
            results = []
            for res in response.points:
                payload = res.payload or {}
                results.append({
                    "id": str(res.id),
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "source_type": payload.get("source_type", ""),
                    "page": payload.get("page"),
                    "total_pages": payload.get("total_pages"),
                    "slide": payload.get("slide"),
                    "slide_title": payload.get("slide_title"),
                    "section": payload.get("section"),
                    "chunk_index": payload.get("chunk_index"),
                    "score": res.score,
                    "latency_stage": {
                        "embed_ms": embed_latency_ms,
                        "qdrant_ms": (time.perf_counter() - t_qdrant_start) * 1000.0,
                    }
                })
            logfire.info(f"✅ Local fallback search succeeded with {len(results)} points.")
            return results
        except Exception as fallback_err:
            logfire.error(f"❌ Local fallback search also failed: {fallback_err}")
            return []
