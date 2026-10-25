import os
import sys
import re
import json
import time
import uuid
import html
from datetime import datetime, timezone
import requests
import streamlit as st
import logfire
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# Initialize Logfire from real configuration
logfire_token = os.getenv("LOGFIRE_TOKEN", "").strip()
if logfire_token:
    try:
        logfire.configure(token=logfire_token)
        LOGFIRE_STATUS = "Tracing enabled (token set)"
    except Exception as e:
        LOGFIRE_STATUS = f"Standby ({e})"
else:
    LOGFIRE_STATUS = None

# Page config
st.set_page_config(
    page_title="TrustRAG",
    page_icon="🛡️",
    layout="wide",
)

# CSS: hide heading anchor action elements only; add bottom padding for thumbs
st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    /* Bottom padding so thumbs buttons clear the chat input */
    [data-testid="stChatMessageContent"] {
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AI_AVATAR = "🛡️"
USER_AVATAR = "👤"
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Snippet truncation helper (Point 2)
# ---------------------------------------------------------------------------

def truncate_snippet(text: str, max_chars: int = 400) -> str:
    """Truncate snippet at a line boundary (max ~max_chars chars).
    If the cut lands inside a ``` code fence, close the fence.
    Appends '...' when truncated.
    """
    if not text or len(text) <= max_chars:
        return text
    # Find the last newline before max_chars
    cut = text.rfind("\n", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    truncated = text[:cut].rstrip()
    # Count open code fences
    fence_count = truncated.count("```")
    if fence_count % 2 == 1:
        truncated += "\n```"
    truncated += "\n..."
    return truncated


# ---------------------------------------------------------------------------
# Feedback helper – single store via /v1/feedback API (Point 9)
# ---------------------------------------------------------------------------

def record_feedback(message_id: str, rating: str, comment: str = ""):
    """Post feedback to the API endpoint (SQLite-backed)."""
    try:
        requests.post(
            f"{BASE_URL}/v1/feedback",
            json={"message_id": message_id, "rating": rating, "comment": comment},
            timeout=5,
        )
    except Exception:
        pass  # Non-critical – don't block UI


@st.cache_data(ttl=60)
def fetch_server_config() -> dict:
    """Fetch live server config from backend for runtime transparency."""
    try:
        resp = requests.get(f"{BASE_URL}/config", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Citation marker helper (Point 5)
# ---------------------------------------------------------------------------

_SOURCE_LINE_RE = re.compile(
    r"\[Source:\s*([^|\]]+?)(?:\s*\|\s*Section:\s*([^\]]+?))?\]\s*(.*)",
    re.DOTALL,
)

def inject_citation_markers(answer: str, sources: list) -> str:
    """Replace (Source: … | Section: …) strings with [n] markers that match the
    source card numbering.  If sources are already in [n] form, return as-is."""
    if not sources or not answer:
        return answer
    # Build lookup: (filename, section) -> card number
    lookup = {}
    for i, src in enumerate(sources):
        if isinstance(src, dict):
            key = (src.get("source", "").strip().lower(), (src.get("section") or "").strip().lower())
        else:
            m = _SOURCE_LINE_RE.match(str(src).strip())
            if m:
                key = (m.group(1).strip().lower(), (m.group(2) or "").strip().lower())
            else:
                continue
        lookup[key] = i + 1

    # Remove inline (Source: … | Section: …) patterns and replace with [n]
    def _repl(m):
        fname = m.group(1).strip().lower()
        sec = (m.group(2) or "").strip().lower()
        n = lookup.get((fname, sec))
        if n:
            return f"[{n}]"
        return m.group(0)

    cleaned = re.sub(
        r"\(Source:\s*([^|)]+?)(?:\s*\|\s*Section:\s*([^)]+?))?\)",
        _repl,
        answer,
    )
    return cleaned


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = f"thread-{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ---------------------------------------------------------------------------
# Sidebar (Point 4 – scope phrase only in main; sidebar keeps tagline)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ TrustRAG")
    st.caption("Grounded answers with sources.")
    st.markdown("---")

    # Logfire badge (Point 7)
    if LOGFIRE_STATUS == "Tracing enabled (token set)":
        st.success(f"Logfire: {LOGFIRE_STATUS}")
    elif LOGFIRE_STATUS:
        st.warning(f"Logfire: {LOGFIRE_STATUS}")

    # Session info (Point 6) – moved here; no full Developer details in sidebar
    with st.expander("📋 Session info", expanded=False):
        st.code(st.session_state.session_id, language="text")
        st.caption("Thread ID")

    # Clear History & Memory with confirm → DELETE /v1/threads (Point 8)
    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear History & Memory", use_container_width=True, type="secondary"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ Clear conversation history and reset memory?")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Yes, Clear", type="primary", use_container_width=True):
                # Call DELETE /v1/threads/{id} to remove server-side state
                try:
                    requests.delete(
                        f"{BASE_URL}/v1/threads/{st.session_state.session_id}",
                        timeout=5,
                    )
                except Exception:
                    pass
                st.session_state.messages = []
                st.session_state.session_id = f"thread-{uuid.uuid4().hex[:12]}"
                st.session_state.feedback_given = {}
                st.session_state.confirm_clear = False
                st.rerun()
        with col_cancel:
            if st.button("Cancel", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


# ---------------------------------------------------------------------------
# Main title + scope phrase (once only – Point 4)
# ---------------------------------------------------------------------------

st.title("🛡️ TrustRAG")
st.caption(
    "Answers grounded in your technical documentation "
    "(Kubernetes, Redis, Databricks, systems and networking), with sources."
)

# ---------------------------------------------------------------------------
# Example question buttons
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "How do you start Redis for a Kubernetes work queue?",
    "What is the difference between HPA and VPA in Kubernetes?",
    "How do you view logs of completed Kubernetes jobs?",
    "What are the three main developer tools for managing Databricks jobs?",
]

selected_example = None
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    cols = st.columns(2)
    for idx, ex in enumerate(EXAMPLE_QUESTIONS):
        with cols[idx % 2]:
            if st.button(ex, key=f"ex_{idx}", use_container_width=True):
                selected_example = ex

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = AI_AVATAR if role == "assistant" else USER_AVATAR
    with st.chat_message(role, avatar=avatar):
        state = message.get("state_type", "normal")

        # Distinct rendering by state (Point 5)
        if state == "blocked":
            st.warning(f"🛡️ **Blocked by Safety Policy**\n\n{message['content']}")
        elif state == "no_context":
            st.info(f"📄 **Not Found in Documentation**\n\n{message['content']}")
        elif state == "rate_limited":
            st.error(f"⏳ **Reasoning Service Rate Limited**\n\n{message['content']}")
            if st.button("🔄 Retry Query", key=f"retry_{idx}"):
                st.session_state.retry_prompt = message.get("question_text")
                st.rerun()
        else:
            # Inject [n] citation markers into answer text (Point 5)
            display_text = inject_citation_markers(
                message["content"], message.get("sources", [])
            )
            st.markdown(display_text)

        # Sources as numbered cards (Point 2 – safe rendering)
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"📚 Sources & Citations ({len(sources)})", expanded=False):
                for s_idx, src in enumerate(sources):
                    if isinstance(src, dict):
                        src_title = src.get("source", "Unknown")
                        section = src.get("section", "General")
                        page = src.get("page")
                        snippet = src.get("snippet", "")
                    else:
                        src_str = str(src).strip()
                        match = _SOURCE_LINE_RE.match(src_str)
                        if match:
                            src_title = match.group(1).strip()
                            section = match.group(2).strip() if match.group(2) else "General"
                            snippet = match.group(3).strip()
                            page = None
                        else:
                            src_title = src_str.split("\n")[0][:60]
                            section = "General"
                            page = None
                            snippet = src_str

                    # Safe truncation (Point 2)
                    snippet = truncate_snippet(snippet, 400)
                    page_badge = f" • p.{page}" if page else ""

                    # Header line using escaped text (no raw HTML from docs)
                    st.markdown(
                        f"**[{s_idx + 1}]** {html.escape(src_title)} · "
                        f"_{html.escape(section)}_{page_badge}"
                    )
                    # Snippet rendered via st.code to avoid HTML injection
                    if snippet:
                        st.code(snippet, language="text")

        # Developer details expander per answer (Point 6)
        if role == "assistant":
            with st.expander("🛠️ Developer details", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Model", message.get("model_used") or "—")
                with c2:
                    lat = message.get("latency_ms")
                    st.metric("Latency", f"{lat} ms" if lat else "—")
                with c3:
                    ttft = message.get("ttft_ms")
                    st.metric("TTFT", f"{ttft} ms" if ttft else "—")
                with c4:
                    st.metric("Retrieval", message.get("retrieval_mode") or "—")

                srv_cfg = fetch_server_config()
                guard_mode_val = message.get("guard_mode") or srv_cfg.get("GUARD_MODE") or "—"
                guard_model_val = message.get("guard_model_used") or srv_cfg.get("GUARDRAIL_MODEL") or "—"

                st.markdown(f"**Thread ID:** `{st.session_state.session_id}`")
                st.markdown(f"**Guard Mode:** `{guard_mode_val}`")
                st.markdown(f"**Guard Model:** `{guard_model_val}`")

                thought = message.get("thought_process", [])
                if thought:
                    st.markdown("**Intent / Planner / Search term:**")
                    for step in thought:
                        st.write(f"• {step}")

        # Thumbs Up / Down → API feedback (Point 9)
        if role == "assistant" and state != "rate_limited":
            fb_key = f"fb_{idx}"
            if fb_key not in st.session_state.feedback_given:
                fcol1, fcol2, _ = st.columns([0.06, 0.06, 0.88])
                with fcol1:
                    if st.button("👍", key=f"up_{idx}", help="Helpful answer"):
                        record_feedback(
                            message_id=message.get("message_id", f"msg-{idx}"),
                            rating="up",
                        )
                        st.session_state.feedback_given[fb_key] = "up"
                        st.toast("Thank you for your feedback! 👍")
                        st.rerun()
                with fcol2:
                    if st.button("👎", key=f"down_{idx}", help="Inaccurate or unhelpful"):
                        record_feedback(
                            message_id=message.get("message_id", f"msg-{idx}"),
                            rating="down",
                        )
                        st.session_state.feedback_given[fb_key] = "down"
                        st.toast("Thank you for your feedback! 👎")
                        st.rerun()
            else:
                rated = st.session_state.feedback_given[fb_key]
                st.caption(
                    f"Feedback: {'👍 Helpful' if rated == 'up' else '👎 Needs improvement'}"
                )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

user_query = None
if st.session_state.retry_prompt:
    user_query = st.session_state.retry_prompt
    st.session_state.retry_prompt = None
elif selected_example:
    user_query = selected_example
else:
    user_query = st.chat_input("Ask about your documentation...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        status_box = st.status("🔍 Initializing TrustRAG pipeline...", expanded=True)
        start_time = time.perf_counter()
        thought_process = []

        meta = {
            "answer_text": "",
            "model_used": None,
            "retrieval_mode": None,
            "sources": [],
            "state_type": "normal",
            "latency_ms": None,
            "ttft_ms": None,
            "message_id": f"msg-{uuid.uuid4().hex[:12]}",
        }

        # Attempt streaming endpoint first (/v1/chat/stream)
        stream_url = f"{BASE_URL}/v1/chat/stream"
        streaming_success = False

        try:
            stream_payload = {
                "q": user_query,
                "thread_id": st.session_state.session_id,
            }
            res = requests.post(stream_url, json=stream_payload, stream=True, timeout=90)

            if res.status_code == 200 and "text/event-stream" in res.headers.get(
                "content-type", ""
            ):
                streaming_success = True
                meta["first_token_seen"] = False

                def token_stream():
                    current_event = None
                    for line in res.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("event:"):
                            current_event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            data_raw = line.split(":", 1)[1].strip()
                            try:
                                data = json.loads(data_raw)
                            except Exception:
                                data = {}

                            if current_event == "status":
                                stage = data.get("stage", "")
                                if stage == "guard":
                                    status_box.update(
                                        label="🛡️ Checking guardrails & safety policies...",
                                        state="running",
                                    )
                                elif stage == "retrieval":
                                    status_box.update(
                                        label="🔎 Retrieving documentation & reranking sources...",
                                        state="running",
                                    )
                                elif stage == "generating":
                                    status_box.update(
                                        label="✍️ Generating grounded answer...",
                                        state="running",
                                    )

                            elif current_event == "sources":
                                meta["sources"] = data

                            elif current_event == "token":
                                delta = data.get("delta", "")
                                if delta and not meta["first_token_seen"]:
                                    meta["first_token_seen"] = True
                                    meta["ttft_ms"] = int(
                                        (time.perf_counter() - start_time) * 1000
                                    )
                                meta["answer_text"] += delta
                                yield delta

                            elif current_event == "done":
                                meta["message_id"] = data.get(
                                    "message_id", meta["message_id"]
                                )
                                meta["model_used"] = data.get("model_used")
                                meta["guard_mode"] = data.get("guard_mode")
                                meta["guard_model_used"] = data.get("guard_model_used")
                                meta["retrieval_mode"] = data.get("retrieval_mode")
                                meta["latency_ms"] = data.get("latency_ms")
                                if data.get("time_to_first_token_ms"):
                                    meta["ttft_ms"] = data["time_to_first_token_ms"]
                                if data.get("status") == "blocked_by_guardrails":
                                    meta["state_type"] = "blocked"

                            elif current_event == "error":
                                err_code = data.get("code", "internal")
                                err_msg = data.get("message", "An error occurred.")
                                if err_code in ("rate_limited", "quota_exhausted"):
                                    meta["state_type"] = "rate_limited"
                                elif err_code == "guard_blocked":
                                    meta["state_type"] = "blocked"
                                elif err_code == "no_context":
                                    meta["state_type"] = "no_context"
                                else:
                                    meta["state_type"] = "error"
                                meta["answer_text"] = err_msg

                st.write_stream(token_stream())

        except Exception:
            streaming_success = False

        # Fallback to /query
        if not streaming_success:
            try:
                status_box.update(label="⚡ Processing...", state="running")
                query_url = f"{BASE_URL}/query"
                q_payload = {"q": user_query, "thread_id": st.session_state.session_id}
                res = requests.post(query_url, json=q_payload, timeout=90)
                data = res.json()

                meta["answer_text"] = data.get("answer", "No response generated.")
                meta["sources"] = data.get("sources", [])
                thought_process = data.get("thought_process", [])
                meta["model_used"] = data.get("model_used")
                meta["guard_mode"] = data.get("guard_mode")
                meta["guard_model_used"] = data.get("guard_model_used")
                meta["retrieval_mode"] = data.get("retrieval_mode")
                meta["latency_ms"] = int((time.perf_counter() - start_time) * 1000)

                status_str = str(data.get("status", "")).lower()
                if "guardrail" in status_str or "blocked" in status_str:
                    meta["state_type"] = "blocked"
                elif "not found" in meta["answer_text"].lower() or "insufficient" in status_str:
                    meta["state_type"] = "no_context"
                elif "rate limit" in meta["answer_text"].lower() or "demand" in meta["answer_text"].lower():
                    meta["state_type"] = "rate_limited"

                st.markdown(meta["answer_text"])
            except Exception as e:
                meta["state_type"] = "rate_limited" if "429" in str(e) else "error"
                meta["answer_text"] = f"Unable to reach reasoning backend: {e}"
                st.error(meta["answer_text"])

        status_box.update(label="✅ Processing Complete", state="complete", expanded=False)

        st.session_state.messages.append({
            "role": "assistant",
            "content": meta["answer_text"],
            "sources": meta["sources"],
            "model_used": meta["model_used"],
            "guard_model_used": meta.get("guard_model_used"),
            "retrieval_mode": meta["retrieval_mode"],
            "thought_process": thought_process,
            "latency_ms": meta["latency_ms"] or int((time.perf_counter() - start_time) * 1000),
            "ttft_ms": meta["ttft_ms"],
            "state_type": meta["state_type"],
            "message_id": meta["message_id"],
            "question_text": user_query,
        })
        st.rerun()
