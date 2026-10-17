import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs hybrid/vector search and semantic reranking for technical queries.
    Preserves document source and section headers in formatted context.
    """
    query = state["current_query"]

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching knowledge base for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_docs = rerank_documents(query, raw_results, top_n=5)
            logfire.info(f"Reranking complete. Kept {len(reranked_docs)} chunks after threshold.")

        formatted_docs = []
        for doc in reranked_docs:
            source = doc.get("source", "Unknown")
            section = doc.get("section") or doc.get("slide_title") or f"Page {doc.get('page')}" if doc.get("page") else "General"
            header = f"[Source: {source} | Section: {section}]"
            formatted_docs.append(f"{header}\n{doc.get('content', '')}")

    status_msg = f"Found {len(formatted_docs)} relevant technical context units." if formatted_docs else "No relevant context found."

    return {
        "documents": formatted_docs,
        "status": status_msg,
        "plan": state["plan"] + (["Context Retrieved"] if formatted_docs else ["Context Filtered Out (Below Threshold)"])
    }
