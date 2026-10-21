# TrustRAG: Modern UI & API Layer Implementation Plan

## 1. Executive Architecture & Goals

The objective is to upgrade TrustRAG from a basic Streamlit prototype into a production-grade enterprise application comprising:
1. **High-Performance Streaming Backend**: FastAPI `/v1/chat/stream` Server-Sent Events (SSE), thread isolation, user feedback, and rate-limiting.
2. **Next.js 14+ App Router Frontend**: Modern React/TypeScript UI with Tailwind CSS, shadcn/ui components, streaming token rendering, interactive citation panels, and dark/light modes.
3. **Packaging & Verification**: Multi-stage Dockerfiles, Docker Compose, automated offline unit/integration test suite, and Playwright smoke testing.

> **Zero Logic Mutation Invariant**:
> Retrieval (`app/services/retrieval/*`), reranking (`FlashRank`), safety guardrails (`app/guardrails/*`), and query planner (`app/agents/nodes/planner.py`) logic remain completely untouched.

---

## 2. API Contract Specification

### 2.1 Server-Sent Events: `POST /v1/chat/stream`
- **Request Headers**:
  - `Content-Type: application/json`
  - `X-API-Key`: Optional API key (when `AUTH_MODE=apikey`)
  - `Cookie: trustrag_user_id=...`: Anonymous session cookie (when `AUTH_MODE=none`)
- **Request Body**:
  ```json
  {
    "q": "What is the difference between HPA and VPA in Kubernetes?",
    "thread_id": "thread-3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "retrieval_mode": "rrf"
  }
  ```
- **Response Headers**:
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`
  - `X-Request-ID`: UUIDv4 tracking ID

- **Event Sequence & Payload Contract**:

| Event Name | Schema Payload | Description / Timing |
| :--- | :--- | :--- |
| `status` | `{"stage": "guard" \| "retrieval" \| "generating"}` | Emitted at each phase transition |
| `sources` | `[{"id": "doc-1", "source": "pods_autoscale.html", "section": "What is HPA?", "page": null, "snippet": "Horizontal Pod Autoscaling..."}]` | Emitted immediately after retrieval & reranking (before token generation); snippet capped at 300 chars |
| `token` | `{"delta": "Based "}` | Emitted incrementally as tokens are streamed from LLM gateway |
| `done` | `{"message_id": "msg-...", "model_used": "openai/gpt-oss-120b" \| null, "retrieval_mode": "rrf", "latency_ms": 1420, "usage": {"prompt_tokens": 850, "completion_tokens": 120, "total_tokens": 970}}` | Emitted at successful completion across all paths (technical, chit-chat, refusal, guard-blocked) |
| `error` | `{"code": "rate_limited", "message": "Reasoning service rate limit reached. Please wait a moment.", "retryable": true}` | Emitted when a caught exception terminates the stream prematurely |

### 2.2 Error Codes Reference
1. `guard_blocked`: Input flagged by safety guardrails (jailbreak, injection, off-topic). Non-retryable.
2. `guard_unavailable`: Guardrail inference service unreachable or timed out. Retryable.
3. `rate_limited`: Provider burst limit (HTTP 429) reached. Retryable.
4. `quota_exhausted`: Daily provider token/request budget exhausted. Non-retryable.
5. `no_context`: Zero relevant documents found and model returned grounded refusal. Non-retryable.
6. `internal`: Uncaught execution or gateway error. Non-retryable (stack traces & keys strictly suppressed).

### 2.3 Feedback API: `POST /v1/feedback`
- **Request Body**:
  ```json
  {
    "message_id": "msg-8f12a34b",
    "rating": "up", // "up" | "down"
    "comment": "Accurate CLI commands provided."
  }
  ```
- **Response**: `{"status": "ok", "feedback_id": "fb-9921"}`
- **Persistence**: Stored in a standalone SQLite database (`feedback.db`, separate from `checkpoints.db`).

### 2.4 Thread Management APIs
- `GET /v1/threads`: List all conversation threads owned by the current authenticated user:
  ```json
  [
    {
      "id": "thread-123",
      "title": "Kubernetes Autoscaling HPA vs VPA",
      "created_at": "2026-10-22T00:15:00Z",
      "updated_at": "2026-10-22T00:16:30Z"
    }
  ]
  ```
- `GET /v1/threads/{id}`: Fetch message history for a specific thread owned by user. Returns 404/403 if thread belongs to another user.
- `DELETE /v1/threads/{id}`: Permanently delete thread history from both thread metadata and SQLite checkpointer.

### 2.5 Authentication & Multi-Tenant Isolation
- `AUTH_MODE=none`: Issues an HTTP-only secure cookie `trustrag_user_id` with a UUIDv4 if absent.
- `AUTH_MODE=apikey`: Validates `X-API-Key` against `API_KEYS` in settings; derives `user_id` from the key identifier.
- `AUTH_MODE=oidc`: Documented stub verifying JWT Bearer tokens; extracts `sub` as `user_id`.
- **Strict Isolation Guarantee**: Every thread is mapped to `owner_user_id` in SQLite. A user attempting to access or write to another user's thread will receive HTTP 403 Forbidden.

---

## 3. Folder Layout

```text
TrustRAG/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py         # POST /v1/chat/stream SSE endpoint
│   │   │   ├── feedback.py     # POST /v1/feedback endpoint
│   │   │   └── threads.py      # GET/DELETE /v1/threads endpoints
│   │   ├── auth.py             # User identity derivation & cookie handling
│   │   ├── errors.py           # Standardized error mapping & exceptions
│   │   └── ratelimit.py        # SlowAPI per-IP/user limiter setup
│   ├── models/
│   │   ├── api_schemas.py      # Pydantic models for v1 API
│   │   └── feedback_db.py      # SQLite storage for feedback & thread metadata
│   ├── main.py                 # FastAPI app mounting v1 routes + legacy /query
│   └── ...                     # (Existing agents, gateway, guardrails untouched)
├── frontend/                   # Next.js 14 App Router project
│   ├── app/
│   │   ├── layout.tsx          # Root layout with theme provider & fonts
│   │   ├── page.tsx            # Main chat interface
│   │   └── globals.css         # Tailwind & theme variables
│   ├── components/
│   │   ├── chat/
│   │   │   ├── message-list.tsx
│   │   │   ├── chat-input.tsx
│   │   │   ├── citation-chip.tsx
│   │   │   ├── citation-sheet.tsx
│   │   │   ├── feedback-buttons.tsx
│   │   │   └── technical-details.tsx
│   │   ├── sidebar/
│   │   │   └── thread-list.tsx
│   │   └── ui/                 # Reusable shadcn/ui components
│   ├── lib/
│   │   ├── api.ts              # SSE client parser & fetch wrappers
│   │   └── utils.ts
│   ├── Dockerfile              # Next.js standalone container
│   ├── .env.example
│   ├── package.json
│   └── tsconfig.json
├── tests/
│   ├── test_chat_stream.py     # Offline SSE order, error code, & refusal tests
│   ├── test_thread_isolation.py# Multi-user thread isolation tests
│   ├── test_feedback.py        # Feedback storage tests
│   └── ...                     # (Existing test suite passing)
├── docker-compose.yml          # Unified orchestrator (api + frontend)
├── Dockerfile                  # Production FastAPI container
└── docs/
    ├── UI_PLAN.md              # Markdown implementation plan
    └── UI_PLAN.html            # Visual interactive HTML plan
```

---

## 4. Key Engineering Risks & Mitigations

| Risk | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **LangGraph Streaming Interoperability** | LangGraph graph execution is synchronous to maintain Logfire tracing context. | Run node pipeline up to retrieval to yield `status` and `sources` events, then invoke gateway LLM with `stream=True` directly in the endpoint while updating the checkpointer state upon completion. |
| **Thread Cross-Contamination** | Attackers or concurrent users guessing `thread_id` UUIDs could inspect other conversations. | Enforce thread ownership mapping table in SQLite. Every thread read/write checks `user_id == thread.owner_id` prior to checkpointer dispatch. |
| **SSE Event Ordering & Buffering** | Reverse proxies (NGINX, Portkey, Cloudflare) may buffer chunks and break real-time token streaming. | Set `X-Accel-Buffering: no` and appropriate chunked transfer headers on all SSE responses. |
| **Feedback / Checkpoint DB Contention** | High feedback volume locking SQLite checkpointer database. | Maintain feedback in a separate SQLite database file (`feedback.db`) with WAL mode enabled. |
| **Secrets & Trace Leakage in Stream Errors** | Raw python tracebacks leaking API keys or internal database paths. | Error boundary intercepts all exceptions and transforms them to sanitized `error` SSE events with mapped error codes. |
