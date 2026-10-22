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

from app.agents.pipeline import execute_rag_pipeline

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
        try:
            generator = execute_rag_pipeline(
                q=q,
                thread_id=thread_id,
                user_id=user_id,
                retrieval_mode=payload.retrieval_mode,
                stream=True,
                guard_fn=guard,
                search_fn=search_enterprise_knowledge,
                rerank_fn=rerank_documents,
                llm_fn=execute_gateway_chat,
            )
            for event_name, data in generator:
                yield format_sse_event(event_name, data)
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
