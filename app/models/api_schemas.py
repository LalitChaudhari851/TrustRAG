from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    q: str = Field(..., description="User query prompt")
    thread_id: Optional[str] = Field(None, description="Conversation thread UUID. Auto-generated if not provided.")
    retrieval_mode: Optional[str] = Field(None, description="Retrieval mode: tinybert | rrf | fusion")


class SourceItem(BaseModel):
    id: str
    source: str
    section: str
    page: Optional[int] = None
    snippet: str = Field(..., max_length=300)


class DoneUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DonePayload(BaseModel):
    message_id: str
    model_used: Optional[str] = None
    retrieval_mode: str
    latency_ms: int
    usage: Dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    code: Literal["guard_blocked", "guard_unavailable", "rate_limited", "quota_exhausted", "no_context", "internal"]
    message: str
    retryable: bool = False


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="ID of the assistant message being rated")
    rating: Literal["up", "down"] = Field(..., description="Feedback rating: up or down")
    comment: Optional[str] = Field(None, description="Optional textual comment or critique")


class FeedbackResponse(BaseModel):
    status: str = "ok"
    feedback_id: str


class ThreadSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ThreadMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    sources: Optional[List[SourceItem]] = None
    created_at: str


class ThreadDetail(BaseModel):
    id: str
    title: str
    messages: List[ThreadMessage]
    created_at: str
    updated_at: str
