import os
import time
import uuid
import logfire
from typing import Optional, Dict, Any, Generator, Tuple

from app.config import settings
from app.guardrails import guard as default_guard
from app.agents.nodes.planner import is_pure_chitchat
from app.services.retrieval.qdrant_service import search_enterprise_knowledge as default_search
from app.services.retrieval.ranking_service import rerank_documents as default_rerank
from app.gateway import execute_gateway_chat as default_execute_chat
from app.models.feedback_db import (
    upsert_thread,
    add_thread_message,
    get_thread_detail,
)

# Shared Grounding Refusal constant
INSUFFICIENT_CONTEXT_REFUSAL = "I cannot answer this question as the required information is not found in the documentation."


def execute_rag_pipeline(
    q: str,
    thread_id: str = "default_user",
    user_id: str = "default_user",
    retrieval_mode: Optional[str] = None,
    stream: bool = False,
    guard_fn=None,
    search_fn=None,
    rerank_fn=None,
    llm_fn=None,
    return_trace: bool = False,
    reranker_margin: Optional[float] = None,
    reranker_min_score: Optional[float] = None,
    candidate_limit: Optional[int] = None,
):
    """
    Unified RAG pipeline shared by both POST /query (stream=False)
    and POST /v1/chat/stream (stream=True).

    Coordinates:
      1. Guardrails check (Gate 1)
      2. Planner (Chitchat vs Technical Knowledge)
      3. Knowledge Retrieval & Multi-mode Reranking (tinybert | rrf | fusion)
      4. Grounded Response Generation (streaming tokens or complete response)
    """
    guard_func = guard_fn or default_guard
    search_func = search_fn or default_search
    rerank_func = rerank_fn or default_rerank
    llm_func = llm_fn or default_execute_chat

    mode_used = (retrieval_mode or getattr(settings, "RETRIEVAL_MODE", "tinybert")).lower()
    start_time = time.perf_counter()

    if not stream:
        # -------------------------------------------------------------
        # SYNCHRONOUS FLOW (POST /query)
        # -------------------------------------------------------------
        # 1. Guardrail Check
        guard_res = guard_func(q)
        rail_fired = guard_res[0] if isinstance(guard_res, tuple) else False
        rail_response = guard_res[1] if isinstance(guard_res, tuple) else None
        guard_model_used = getattr(guard_res, "guard_model_used", None) or getattr(settings, "GUARDRAIL_MODEL", "openai/gpt-oss-20b")

        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            res = {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "blocked_by_guardrails",
                "sources": [],
                "retrieval_mode": mode_used,
                "model_used": None,
                "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
                "guard_model_used": guard_model_used,
            }
            if return_trace:
                res["retrieval_trace"] = []
                res["context_sent_to_llm"] = ""
            return res

        # 2. Check for Social Chit-Chat
        if is_pure_chitchat(q):
            logfire.info(f"⚡ Pre-router identified pure chitchat: '{q}'")
            prompt = f"""
You are a friendly and helpful Enterprise AI Assistant named TrustRAG.
Answer the user's latest message politely and concisely.

USER QUESTION:
"{q}"
"""
            resp = llm_func(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                stream=False,
            )
            content = resp.choices[0].message.content if getattr(resp, "choices", None) else ""
            model_returned = getattr(resp, "model", None) or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
            return {
                "question": q,
                "answer": content,
                "thought_process": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
                "status": "Handling conversationally (chitchat/memory)...",
                "sources": [],
                "retrieval_mode": mode_used,
                "model_used": model_returned,
                "guard_model_used": guard_model_used,
            }

        # 3. Knowledge Retrieval & Reranking
        cand_pool_limit = candidate_limit if candidate_limit is not None else 15
        raw_results = search_func(q, limit=cand_pool_limit)
        if mode_used == "rrf":
            selected_docs = raw_results[:5]
        elif mode_used == "fusion":
            reranked = rerank_func(q, raw_results, top_n=5, min_score=reranker_min_score, relative_margin=reranker_margin)
            seen = set()
            fused = []
            for d in reranked[:3]:
                k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
                if k not in seen:
                    seen.add(k)
                    fused.append(d)
            for d in raw_results[:3]:
                k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
                if k not in seen:
                    seen.add(k)
                    fused.append(d)
            selected_docs = fused[:6]
        else:
            selected_docs = rerank_func(q, raw_results, top_n=5, min_score=reranker_min_score, relative_margin=reranker_margin)

        if not selected_docs:
            res = {
                "question": q,
                "answer": INSUFFICIENT_CONTEXT_REFUSAL,
                "thought_process": ["Intent: Technical", "Refusal: Insufficient Context"],
                "status": "Insufficient context.",
                "sources": [],
                "retrieval_mode": mode_used,
                "model_used": None,
                "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
                "guard_model_used": guard_model_used,
            }
            if return_trace:
                res["retrieval_trace"] = []
                res["raw_search_count"] = len(raw_results)
                res["candidate_pool_limit"] = cand_pool_limit
                res["context_sent_to_llm"] = ""
                res["effective_reranker_margin"] = reranker_margin if reranker_margin is not None else getattr(settings, "RERANKER_RELATIVE_MARGIN", 0.20)
                res["effective_reranker_min_score"] = reranker_min_score if reranker_min_score is not None else getattr(settings, "RERANKER_MIN_SCORE", 0.08)
            return res

        formatted_context = ""
        sources_list = []
        for doc in selected_docs:
            source = doc.get("source", "Unknown")
            page_val = f"Page {doc.get('page')}" if doc.get("page") else None
            section = doc.get("section") or doc.get("slide_title") or page_val or "General"
            header = f"[Source: {source} | Section: {section}]"
            sources_list.append(f"{header}\n{doc.get('content', '')}")
            formatted_context += f"{header}\n{doc.get('content', '')}\n\n"

        # 4. Response Generation
        prompt = f"""
You are a Senior Technical Architect answering questions with strict factual grounding.
Answer the user question strictly using ONLY the facts directly stated in the TECHNICAL CONTEXT provided below.

CRITICAL GROUNDING RULES:
1. Answer ONLY from the provided TECHNICAL CONTEXT. Do not speculate, extrapolate, or use outside knowledge.
2. If the TECHNICAL CONTEXT does not contain the specific facts needed to answer the question, respond:
   "I cannot answer this question as the required information is not found in the documentation."
3. Do NOT invent flags, parameters, YAML fields, or command syntax not shown in the context.

TECHNICAL CONTEXT:
{formatted_context}

USER QUESTION:
"{q}"
"""
        resp = llm_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            stream=False,
        )
        content = resp.choices[0].message.content if getattr(resp, "choices", None) else ""
        model_returned = getattr(resp, "model", None) or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")

        trace_data = [
            {
                "id": str(doc.get("id")),
                "source": doc.get("source"),
                "section": doc.get("section") or doc.get("slide_title"),
                "chunk_index": doc.get("chunk_index"),
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "content": doc.get("content", ""),
            }
            for doc in selected_docs
        ]

        res = {
            "question": q,
            "answer": content,
            "thought_process": ["Intent: Technical", f"Search Term: {q}", "Context Retrieved"],
            "status": f"Found {len(selected_docs)} relevant technical context units.",
            "sources": sources_list,
            "retrieval_mode": mode_used,
            "model_used": model_returned,
            "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
            "guard_model_used": guard_model_used,
        }
        if return_trace:
            res["retrieval_trace"] = trace_data
            res["context_sent_to_llm"] = formatted_context
            res["raw_search_count"] = len(raw_results)
            res["candidate_pool_limit"] = cand_pool_limit
            res["effective_reranker_margin"] = reranker_margin if reranker_margin is not None else getattr(settings, "RERANKER_RELATIVE_MARGIN", 0.20)
            res["effective_reranker_min_score"] = reranker_min_score if reranker_min_score is not None else getattr(settings, "RERANKER_MIN_SCORE", 0.08)
        return res

    # -----------------------------------------------------------------
    # STREAMING FLOW (POST /v1/chat/stream)
    # Generator yielding tuples: (event_name, data_payload)
    # -----------------------------------------------------------------
    def stream_generator() -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        ttft_time = None
        accumulated_text = ""
        model_used = None

        # 1. Guardrail Stage
        yield ("status", {"stage": "guard"})
        guard_res = guard_func(q)
        rail_fired = guard_res[0] if isinstance(guard_res, tuple) else False
        rail_response = guard_res[1] if isinstance(guard_res, tuple) else None
        guard_model_used = getattr(guard_res, "guard_model_used", None) or getattr(settings, "GUARDRAIL_MODEL", "openai/gpt-oss-20b")

        if rail_fired:
            logfire.info(f"🛡️ Guardrail blocked request | thread={thread_id} | user={user_id}")
            add_thread_message(thread_id, user_id, "user", q)
            asst_msg_id = add_thread_message(thread_id, user_id, "assistant", rail_response)

            yield ("token", {"delta": rail_response})
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            yield ("done", {
                "message_id": asst_msg_id,
                "status": "blocked_by_guardrails",
                "model_used": None,
                "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
                "guard_model_used": guard_model_used,
                "retrieval_mode": mode_used,
                "latency_ms": latency_ms,
                "time_to_first_token_ms": latency_ms,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            return

        # Save user prompt
        add_thread_message(thread_id, user_id, "user", q)

        # Retrieve conversation history
        thread_data = get_thread_detail(thread_id, user_id)
        history_msgs = thread_data.get("messages", [])[:-1] if thread_data else []
        history_str = ""
        for m in history_msgs[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            history_str += f"{role}: {m['content']}\n"

        # 2. Check for Social Chit-Chat / Meta Queries
        if is_pure_chitchat(q):
            yield ("status", {"stage": "generating"})
            prompt = f"""
You are a friendly and helpful Enterprise AI Assistant named TrustRAG.
Answer the user's latest message politely and concisely.

CONVERSATION HISTORY:
{history_str}

LATEST MESSAGE:
"{q}"
"""
            stream_resp = llm_func(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                stream=True,
            )

            for chunk in stream_resp:
                chunk_model = getattr(chunk, "model", None)
                if chunk_model and not model_used:
                    model_used = chunk_model
                choices = getattr(chunk, "choices", [])
                if choices:
                    delta = choices[0].delta.content or ""
                    if delta:
                        if ttft_time is None:
                            ttft_time = int((time.perf_counter() - start_time) * 1000)
                        accumulated_text += delta
                        yield ("token", {"delta": delta})

            model_used = model_used or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
            asst_msg_id = add_thread_message(thread_id, user_id, "assistant", accumulated_text)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            yield ("done", {
                "message_id": asst_msg_id,
                "model_used": model_used,
                "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
                "guard_model_used": guard_model_used,
                "retrieval_mode": mode_used,
                "latency_ms": latency_ms,
                "time_to_first_token_ms": ttft_time,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(accumulated_text.split()),
                    "total_tokens": len(prompt.split()) + len(accumulated_text.split()),
                },
            })
            return

        # 3. Retrieval Stage
        yield ("status", {"stage": "retrieval"})
        raw_results = search_func(q, limit=15)

        if mode_used == "rrf":
            selected_docs = raw_results[:5]
        elif mode_used == "fusion":
            reranked = rerank_func(q, raw_results, top_n=5, min_score=reranker_min_score, relative_margin=reranker_margin)
            seen = set()
            fused = []
            for d in reranked[:3]:
                k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
                if k not in seen:
                    seen.add(k)
                    fused.append(d)
            for d in raw_results[:3]:
                k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
                if k not in seen:
                    seen.add(k)
                    fused.append(d)
            selected_docs = fused[:6]
        else:
            selected_docs = rerank_func(q, raw_results, top_n=5, min_score=reranker_min_score, relative_margin=reranker_margin)

        # Format and emit sources
        sources_payload = []
        formatted_context = ""
        for i, doc in enumerate(selected_docs):
            source_name = doc.get("source", "Unknown")
            page_val = doc.get("page")
            page_str = f"Page {page_val}" if page_val else None
            section_val = doc.get("section") or doc.get("slide_title") or page_str or "General"
            content_text = doc.get("content", "")
            snippet_text = content_text[:300].strip()

            sources_payload.append({
                "id": f"doc-{i+1}",
                "source": source_name,
                "section": section_val,
                "page": page_val,
                "snippet": snippet_text,
            })
            formatted_context += f"[Source: {source_name} | Section: {section_val}]\n{content_text}\n\n"

        yield ("sources", sources_payload)

        trace_payload = [
            {
                "id": str(doc.get("id")),
                "source": doc.get("source"),
                "section": doc.get("section") or doc.get("slide_title"),
                "chunk_index": doc.get("chunk_index"),
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "content": doc.get("content", ""),
            }
            for doc in selected_docs
        ]
        if return_trace:
            yield ("trace", {"retrieved_docs": trace_payload, "context_sent_to_llm": formatted_context})

        # Check if retrieval yielded sufficient context
        if not selected_docs:
            yield ("token", {"delta": INSUFFICIENT_CONTEXT_REFUSAL})
            asst_msg_id = add_thread_message(thread_id, user_id, "assistant", INSUFFICIENT_CONTEXT_REFUSAL, sources=sources_payload)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            yield ("done", {
                "message_id": asst_msg_id,
                "model_used": None,
                "guard_model_used": guard_model_used,
                "retrieval_mode": mode_used,
                "latency_ms": latency_ms,
                "time_to_first_token_ms": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            return

        # 4. Generating Stage
        yield ("status", {"stage": "generating"})
        prompt = f"""
You are a Senior Technical Architect answering questions with strict factual grounding.
Answer the user question strictly using ONLY the facts directly stated in the TECHNICAL CONTEXT provided below.

CRITICAL GROUNDING RULES:
1. Answer ONLY from the provided TECHNICAL CONTEXT. Do not speculate, extrapolate, or use outside knowledge.
2. If the TECHNICAL CONTEXT does not contain the specific facts needed to answer the question, respond:
   "I cannot answer this question as the required information is not found in the documentation."
3. Do NOT invent flags, parameters, YAML fields, or command syntax not shown in the context.

TECHNICAL CONTEXT:
{formatted_context}

CONVERSATION HISTORY:
{history_str}

USER QUESTION:
"{q}"
"""
        stream_resp = llm_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            stream=True,
        )

        for chunk in stream_resp:
            chunk_model = getattr(chunk, "model", None)
            if chunk_model and not model_used:
                model_used = chunk_model
            choices = getattr(chunk, "choices", [])
            if choices:
                delta = choices[0].delta.content or ""
                if delta:
                    if ttft_time is None:
                        ttft_time = int((time.perf_counter() - start_time) * 1000)
                    accumulated_text += delta
                    yield ("token", {"delta": delta})

        model_used = model_used or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
        asst_msg_id = add_thread_message(thread_id, user_id, "assistant", accumulated_text, sources=sources_payload)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        yield ("done", {
            "message_id": asst_msg_id,
            "model_used": model_used,
            "guard_mode": getattr(settings, "GUARD_MODE", "single_call"),
            "guard_model_used": guard_model_used,
            "retrieval_mode": mode_used,
            "latency_ms": latency_ms,
            "time_to_first_token_ms": ttft_time,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(accumulated_text.split()),
                "total_tokens": len(prompt.split()) + len(accumulated_text.split()),
            },
        })

    return stream_generator()
