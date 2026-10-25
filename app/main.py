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


def check_models_availability():
    warnings = []
    available = []
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        available = [m.id for m in client.models.list().data]
        if settings.GROQ_MODEL not in available:
            msg = f"Configured GROQ_MODEL '{settings.GROQ_MODEL}' is not available on Groq account! Available: {sorted(available)}"
            logfire.warning(f"⚠️ {msg}")
            warnings.append(msg)
        if settings.GUARDRAIL_MODEL not in available:
            msg = f"Configured GUARDRAIL_MODEL '{settings.GUARDRAIL_MODEL}' is not available on Groq account! Available: {sorted(available)}"
            logfire.warning(f"⚠️ {msg}")
            warnings.append(msg)
    except Exception as e:
        msg = f"Could not query Groq models list at startup: {e}"
        logfire.warning(f"⚠️ {msg}")
        warnings.append(msg)
    return {
        "available_models": available,
        "warnings": warnings,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Log guard configuration and validate models against provider
    CODE_DEFAULT_GUARD_MODE = "single_call"
    active_guard_mode = getattr(settings, "GUARD_MODE", "single_call")
    active_guard_model = getattr(settings, "GUARDRAIL_MODEL", "openai/gpt-oss-20b")
    logfire.info(
        f"🛡️ Startup Guardrail Config: GUARD_MODE='{active_guard_mode}', GUARDRAIL_MODEL='{active_guard_model}'"
    )
    if active_guard_mode != CODE_DEFAULT_GUARD_MODE:
        logfire.warning(
            f"⚠️ Active GUARD_MODE '{active_guard_mode}' differs from code default '{CODE_DEFAULT_GUARD_MODE}'"
        )
    app.state.model_check = check_models_availability()
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
    from app.guardrails.single_call import get_active_guard_model
    model_check = getattr(app.state, "model_check", {}) if hasattr(app, "state") else {}
    return {
        "GROQ_MODEL": settings.GROQ_MODEL,
        "GUARD_MODE": settings.GUARD_MODE,
        "code_default_guard_mode": "single_call",
        "GUARDRAIL_MODEL": settings.GUARDRAIL_MODEL,
        "guard_model_used": get_active_guard_model(),
        "RETRIEVAL_MODE": settings.RETRIEVAL_MODE,
        "QUERY_REWRITE": settings.QUERY_REWRITE,
        "model_warnings": model_check.get("warnings", []),
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
    
    
from app.agents.pipeline import execute_rag_pipeline


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the shared LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    try:
        return execute_rag_pipeline(
            q=q,
            thread_id=thread_id,
            retrieval_mode=request.retrieval_mode,
            stream=False,
            guard_fn=guard,
        )
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
