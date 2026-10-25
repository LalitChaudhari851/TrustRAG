import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag_v2")
    DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "all-mpnet-base-v2")
    HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
    RERANKER_MIN_SCORE = float(os.getenv("RERANKER_MIN_SCORE", "0.08"))
    RERANKER_RELATIVE_MARGIN = float(os.getenv("RERANKER_RELATIVE_MARGIN", "0.20"))
    PROJECT_UUID_NAMESPACE = os.getenv("PROJECT_UUID_NAMESPACE", "e38b3687-9418-4b7d-b6a4-22b6478d1326")

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "openai/gpt-oss-20b")
    GUARD_MODE = os.getenv("GUARD_MODE", "single_call").lower()  # single_call | nemo
    QUERY_REWRITE = os.getenv("QUERY_REWRITE", "followup_only").lower()  # followup_only | always | off
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "tinybert").lower()  # tinybert | rrf | fusion


    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "brag")

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # --- PERSISTENCE & CHECKPOINTING ---
    DATABASE_URL = os.getenv("DATABASE_URL")
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.db")

    # --- GATEWAY RESILIENCE (TENACITY) ---
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_MIN_WAIT = float(os.getenv("LLM_RETRY_MIN_WAIT", "2.0"))
    LLM_RETRY_MAX_WAIT = float(os.getenv("LLM_RETRY_MAX_WAIT", "10.0"))

    # --- PRODUCT & API LAYER ---
    AUTH_MODE = os.getenv("AUTH_MODE", "none").lower()  # none | apikey | oidc
    API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
    CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,*").split(",") if o.strip()]
    FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "feedback.db")
    RATE_LIMIT_PER_MINUTE = os.getenv("RATE_LIMIT_PER_MINUTE", "60/minute")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()

