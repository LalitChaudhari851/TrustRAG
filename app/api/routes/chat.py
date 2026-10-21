import time
import uuid
import json
import logfire
from typing import AsyncGenerator
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.api_schemas import ChatStreamRequest
from app.models.feedback_db import (
    upsert_thread,
    add_thread_message,
    get_thread_detail,
)
from app.api.auth import get_current_user
from app.api.errors import map_exception_to_error_event, format_sse_event
from app.api.ratelimit import limiter
from app.guardrails import guard
from app.agents.nodes.planner import is_pure_chitchat
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.gateway import execute_gateway_chat

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/chat/stream")
@limiter.limit(getattr(settings, "RATE_LIMIT_PER_MINUTE", "60/minute"))
def chat_stream(payload: ChatStreamRequest, request: Request, response: Response):
    """
    Server-Sent Events (SSE) streaming chat endpoint.
    Emits events:
      - status {stage: guard|retrieval|generating}
      - sources [{id, source, section, page, snippet(<=300 chars)}]
      - token {delta}
      - done {message_id, model_used, retrieval_mode, latency_ms, usage}
      - error {code, message, retryable}
    """
    user_id = get_current_user(request, response)
    thread_id = payload.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    q = (payload.q or "").strip()

    # Verify thread ownership before initiating stream
    if payload.thread_id:
        try:
            upsert_thread(thread_id, user_id)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: you do not own this thread."
            )
    else:
        upsert_thread(thread_id, user_id, title=q[:40].replace("\n", " ") or "New Conversation")

    request_id = str(uuid.uuid4())

    def event_generator():
        start_time = time.perf_counter()
        mode_used = (payload.retrieval_mode or getattr(settings, "RETRIEVAL_MODE", "tinybert")).lower()
        accumulated_text = ""
        model_used = None

        try:
            # 1. Guardrail Stage
            yield format_sse_event("status", {"stage": "guard"})
            
            rail_fired, rail_response = guard(q)
            if rail_fired:
                logfire.info(f"🛡️ Guardrail blocked request | thread={thread_id} | user={user_id}")
                # Save messages to thread history
                add_thread_message(thread_id, user_id, "user", q)
                asst_msg_id = add_thread_message(thread_id, user_id, "assistant", rail_response)

                # Stream rejection message token
                yield format_sse_event("token", {"delta": rail_response})
                
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                yield format_sse_event("done", {
                    "message_id": asst_msg_id,
                    "model_used": None,
                    "retrieval_mode": mode_used,
                    "latency_ms": latency_ms,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
                yield format_sse_event("status", {"stage": "generating"})
                
                prompt = f"""
You are a friendly and helpful Enterprise AI Assistant named TrustRAG.
Answer the user's latest message politely and concisely.

CONVERSATION HISTORY:
{history_str}

LATEST MESSAGE:
"{q}"
"""
                stream_resp = execute_gateway_chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200,
                    stream=True
                )

                for chunk in stream_resp:
                    chunk_model = getattr(chunk, "model", None)
                    if chunk_model and not model_used:
                        model_used = chunk_model
                    choices = getattr(chunk, "choices", [])
                    if choices:
                        delta = choices[0].delta.content or ""
                        if delta:
                            accumulated_text += delta
                            yield format_sse_event("token", {"delta": delta})

                model_used = model_used or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
                asst_msg_id = add_thread_message(thread_id, user_id, "assistant", accumulated_text)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                yield format_sse_event("done", {
                    "message_id": asst_msg_id,
                    "model_used": model_used,
                    "retrieval_mode": mode_used,
                    "latency_ms": latency_ms,
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(accumulated_text.split()),
                        "total_tokens": len(prompt.split()) + len(accumulated_text.split())
                    }
                })
                return

            # 3. Retrieval Stage
            yield format_sse_event("status", {"stage": "retrieval"})
            raw_results = search_enterprise_knowledge(q, limit=15)

            if mode_used == "rrf":
                selected_docs = raw_results[:5]
            elif mode_used == "fusion":
                reranked = rerank_documents(q, raw_results, top_n=5)
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
                # Default: tinybert
                selected_docs = rerank_documents(q, raw_results, top_n=5)

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
                    "snippet": snippet_text
                })
                formatted_context += f"[Source: {source_name} | Section: {section_val}]\n{content_text}\n\n"

            yield format_sse_event("sources", sources_payload)

            # Check if retrieval yielded sufficient context
            if not selected_docs:
                refusal = "I cannot answer this question as the required information is not found in the documentation."
                yield format_sse_event("token", {"delta": refusal})
                asst_msg_id = add_thread_message(thread_id, user_id, "assistant", refusal, sources=sources_payload)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                yield format_sse_event("done", {
                    "message_id": asst_msg_id,
                    "model_used": None,
                    "retrieval_mode": mode_used,
                    "latency_ms": latency_ms,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                })
                return

            # 4. Generating Stage
            yield format_sse_event("status", {"stage": "generating"})
            
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
            stream_resp = execute_gateway_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                stream=True
            )

            for chunk in stream_resp:
                chunk_model = getattr(chunk, "model", None)
                if chunk_model and not model_used:
                    model_used = chunk_model
                choices = getattr(chunk, "choices", [])
                if choices:
                    delta = choices[0].delta.content or ""
                    if delta:
                        accumulated_text += delta
                        yield format_sse_event("token", {"delta": delta})

            model_used = model_used or getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
            asst_msg_id = add_thread_message(thread_id, user_id, "assistant", accumulated_text, sources=sources_payload)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            yield format_sse_event("done", {
                "message_id": asst_msg_id,
                "model_used": model_used,
                "retrieval_mode": mode_used,
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(accumulated_text.split()),
                    "total_tokens": len(prompt.split()) + len(accumulated_text.split())
                }
            })

        except Exception as exc:
            logfire.error(f"❌ Error during /v1/chat/stream: {exc}")
            err = map_exception_to_error_event(exc)
            yield format_sse_event("error", err)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id
        }
    )
