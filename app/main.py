# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
import uuid
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.config import settings
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard
from app.services.retrieval.embedding import _init as init_embeddings
from app.agents.persistence import get_checkpointer, close_checkpointer
from app.models.feedback_db import init_feedback_db
from app.api.ratelimit import limiter

from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize rails, warm up embeddings, SQLite checkpointer & feedback DB
    initialize_rails()
    init_embeddings()
    get_checkpointer()
    init_feedback_db()
    yield
    # Shutdown: Cleanly close persistent connection pools
    close_checkpointer()


# Initialize FastAPI with Lifespan
app = FastAPI(title="TrustRAG Enterprise API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Include v1 Modern Product API Routes
from app.api.routes.chat import router as chat_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.threads import router as threads_router

app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(threads_router)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    retrieval_mode: Optional[str] = None
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/config")
def get_server_config():
    """Returns the live active runtime configuration for audit and eval verification."""
    from app.config import settings
    return {
        "GROQ_MODEL": settings.GROQ_MODEL,
        "GUARD_MODE": settings.GUARD_MODE,
        "GUARDRAIL_MODEL": settings.GUARDRAIL_MODEL,
        "RETRIEVAL_MODE": settings.RETRIEVAL_MODE,
        "QUERY_REWRITE": settings.QUERY_REWRITE,
    }


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "retrieval_mode": request.retrieval_mode,
        "model_used": None
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "model_used": None
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        mode_used = (request.retrieval_mode or getattr(settings, "RETRIEVAL_MODE", "tinybert")).lower()
        logfire.info(f"Retrieval mode used: {mode_used}")
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "retrieval_mode": mode_used,
            "model_used": final_output.get("model_used")
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "model_used": None
        }
