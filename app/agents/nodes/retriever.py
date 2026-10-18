import logfire
from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs hybrid/vector search and semantic reranking for technical queries.
    Supports RETRIEVAL_MODE=tinybert|rrf|fusion.
    Preserves document source and section headers in formatted context.
    """
    query = state["current_query"]

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching knowledge base for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        mode = (state.get("retrieval_mode") or getattr(settings, "RETRIEVAL_MODE", "tinybert")).lower()

        if mode == "rrf":
            # Pure RRF from Qdrant without TinyBERT reranking
            selected_docs = raw_results[:5]
            logfire.info(f"RETRIEVAL_MODE=rrf: kept top {len(selected_docs)} RRF candidates.")
        elif mode == "fusion":
            # Union of top-3 RRF-order and top-3 TinyBERT-reranked chunks, cap 6, de-duplicated
            with logfire.span("⚖️ Semantic Reranking"):
                reranked_docs = rerank_documents(query, raw_results, top_n=5)

            top_rrf = raw_results[:3]
            top_rerank = reranked_docs[:3]

            seen_keys = set()
            fused_docs = []

            def get_doc_key(d):
                return (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])

            # Prioritize top TinyBERT chunks
            for d in top_rerank:
                k = get_doc_key(d)
                if k not in seen_keys:
                    seen_keys.add(k)
                    fused_docs.append(d)

            # Union with top RRF chunks
            for d in top_rrf:
                k = get_doc_key(d)
                if k not in seen_keys:
                    seen_keys.add(k)
                    fused_docs.append(d)
                    if len(fused_docs) >= 6:
                        break

            selected_docs = fused_docs[:6]
            logfire.info(f"RETRIEVAL_MODE=fusion: fused {len(selected_docs)} chunks (top-3 RRF + top-3 TinyBERT, deduplicated, cap 6).")
        else:
            # Default: tinybert
            with logfire.span("⚖️ Semantic Reranking"):
                selected_docs = rerank_documents(query, raw_results, top_n=5)
                logfire.info(f"Reranking complete. Kept {len(selected_docs)} chunks after threshold.")

        formatted_docs = []
        for doc in selected_docs:
            source = doc.get("source", "Unknown")
            section = doc.get("section") or doc.get("slide_title") or f"Page {doc.get('page')}" if doc.get("page") else "General"
            header = f"[Source: {source} | Section: {section}]"
            formatted_docs.append(f"{header}\n{doc.get('content', '')}")

    status_msg = f"Found {len(formatted_docs)} relevant technical context units." if formatted_docs else "No relevant context found."

    return {
        "documents": formatted_docs,
        "status": status_msg,
        "plan": state.get("plan", []) + (["Context Retrieved"] if formatted_docs else ["Context Filtered Out (Below Threshold)"])
    }
