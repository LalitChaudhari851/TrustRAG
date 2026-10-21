import os
import sys
import re
import json
import time
import uuid
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

# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    logfire.configure(token=token)
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    LOGFIRE_STATUS = f"Standby ({e})"

# 1. ONE PRODUCT NAME IN PAGE CONFIG
st.set_page_config(
    page_title="TrustRAG",
    page_icon="🛡️",
    layout="wide",
)

AI_AVATAR = "🛡️"
USER_AVATAR = "👤"
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- FEEDBACK HELPER (Point 6) ---
FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feedback.jsonl"))

def record_feedback(message_id: str, session_id: str, question: str, answer: str, rating: str):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": message_id,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": rating
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = f"thread-{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None


# 1. ONE PRODUCT NAME IN SIDEBAR
with st.sidebar:
    st.title("🛡️ TrustRAG")
    st.caption("Enterprise Agentic RAG Platform")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    
    # Point 3: Memory ID moved inside Developer details in sidebar
    with st.expander("🛠️ Developer details", expanded=False):
        st.code(st.session_state.session_id, language="text")
        st.caption("Active Session Thread ID")

    if st.button("🗑️ Clear History & Memory", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.session_id = f"thread-{uuid.uuid4().hex[:12]}"
        st.session_state.feedback_given = {}
        st.rerun()


# 1. ONE PRODUCT NAME IN MAIN TITLE
st.title("🛡️ TrustRAG")
st.caption("Verified enterprise assistant for Kubernetes, Intel hardware, and distributed systems.")

# Point 8: 4 EXAMPLE-QUESTION BUTTONS
EXAMPLE_QUESTIONS = [
    "How do you start Redis for a Kubernetes work queue?",
    "What is the difference between HPA and VPA in Kubernetes?",
    "How do you view logs of completed Kubernetes jobs?",
    "What are the three main developer tools for managing Databricks jobs?"
]

selected_example = None
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    cols = st.columns(2)
    for idx, ex in enumerate(EXAMPLE_QUESTIONS):
        with cols[idx % 2]:
            if st.button(ex, key=f"ex_{idx}", use_container_width=True):
                selected_example = ex

# Point 5: Render Chat History with Distinct States and Feedback
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = AI_AVATAR if role == "assistant" else USER_AVATAR
    with st.chat_message(role, avatar=avatar):
        state = message.get("state_type", "normal")
        
        # Point 5: Distinct rendering by state
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
            st.markdown(message["content"])

        # Point 4: Sources as numbered cards
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
                        match = re.match(r"^\[Source:\s*([^|\]]+)(?:\s*\|\s*Section:\s*([^\]]+))?\]\s*(.*)", src_str, re.DOTALL)
                        if match:
                            src_title = match.group(1).strip()
                            section = match.group(2).strip() if match.group(2) else "General"
                            snippet = match.group(3).strip()[:300]
                            page = None
                        else:
                            src_title = src_str.split("\n")[0][:60]
                            section = "General"
                            page = None
                            snippet = src_str[:300]

                    page_badge = f" | Page {page}" if page else ""
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #374151; background: #111827; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                            <div style="font-weight: 600; color: #60a5fa; font-size: 0.9rem;">
                                [{s_idx + 1}] {src_title} &bull; <span style="color: #9ca3af;">{section}</span>{page_badge}
                            </div>
                            <div style="color: #d1d5db; font-size: 0.82rem; margin-top: 6px; line-height: 1.4;">
                                {snippet}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Points 3 & 7: Developer details expander (collapsed by default)
        if role == "assistant":
            with st.expander("🛠️ Developer details", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Used", message.get("model_used") or "None")
                with col2:
                    lat = message.get("latency_ms")
                    st.metric("Latency", f"{lat} ms" if lat else "N/A")
                with col3:
                    st.metric("Retrieval Mode", message.get("retrieval_mode") or "None")

                st.markdown(f"**Memory ID:** `{st.session_state.session_id}`")

                thought = message.get("thought_process", [])
                if thought:
                    st.markdown("**Intent, Planner & Reasoning Steps:**")
                    for step in thought:
                        st.write(f"&bull; {step}")

        # Point 6: Thumbs Up / Down per answer -> feedback.jsonl
        if role == "assistant" and state != "rate_limited":
            fb_key = f"fb_{idx}"
            if fb_key not in st.session_state.feedback_given:
                fcol1, fcol2, _ = st.columns([0.08, 0.08, 0.84])
                with fcol1:
                    if st.button("👍", key=f"up_{idx}", help="Helpful answer"):
                        record_feedback(
                            message_id=message.get("message_id", f"msg-{idx}"),
                            session_id=st.session_state.session_id,
                            question=message.get("question_text", ""),
                            answer=message["content"],
                            rating="up"
                        )
                        st.session_state.feedback_given[fb_key] = "up"
                        st.toast("Thank you for your feedback! 👍")
                        st.rerun()
                with fcol2:
                    if st.button("👎", key=f"down_{idx}", help="Inaccurate or unhelpful"):
                        record_feedback(
                            message_id=message.get("message_id", f"msg-{idx}"),
                            session_id=st.session_state.session_id,
                            question=message.get("question_text", ""),
                            answer=message["content"],
                            rating="down"
                        )
                        st.session_state.feedback_given[fb_key] = "down"
                        st.toast("Thank you for your feedback! 👎")
                        st.rerun()
            else:
                rated = st.session_state.feedback_given[fb_key]
                st.caption(f"Feedback submitted: {'👍 Helpful' if rated == 'up' else '👎 Needs improvement'}")


# --- INPUT HANDLING ---
user_query = None
if st.session_state.retry_prompt:
    user_query = st.session_state.retry_prompt
    st.session_state.retry_prompt = None
elif selected_example:
    user_query = selected_example
else:
    user_query = st.chat_input("Ask about Kubernetes, Redis, or Databricks documentation...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        # Point 2: Stage indicator spinner (guard -> retrieval -> generating)
        status_box = st.status("🔍 Initializing TrustRAG pipeline...", expanded=True)
        
        meta = {
            "answer_text": "",
            "model_used": None,
            "retrieval_mode": None,
            "sources": [],
            "state_type": "normal",
            "latency_ms": None,
            "message_id": f"msg-{uuid.uuid4().hex[:12]}"
        }

        # Point 2: Attempt streaming endpoint first (/v1/chat/stream)
        stream_url = f"{BASE_URL}/v1/chat/stream"
        streaming_success = False

        try:
            stream_payload = {
                "q": user_query,
                "thread_id": st.session_state.session_id
            }
            res = requests.post(stream_url, json=stream_payload, stream=True, timeout=90)
            
            if res.status_code == 200 and "text/event-stream" in res.headers.get("content-type", ""):
                streaming_success = True
                
                # Generator for st.write_stream
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

                            # Stage updates (Point 2)
                            if current_event == "status":
                                stage = data.get("stage", "")
                                if stage == "guard":
                                    status_box.update(label="🛡️ Stage 1/3: Safety Guardrails check...", state="running")
                                elif stage == "retrieval":
                                    status_box.update(label="🔎 Stage 2/3: Hybrid Retrieval & FlashRank reranking...", state="running")
                                elif stage == "generating":
                                    status_box.update(label="✍️ Stage 3/3: Grounded Answer Synthesis...", state="running")
                            
                            elif current_event == "sources":
                                meta["sources"] = data
                            
                            elif current_event == "token":
                                delta = data.get("delta", "")
                                meta["answer_text"] += delta
                                yield delta

                            elif current_event == "done":
                                meta["message_id"] = data.get("message_id", meta["message_id"])
                                meta["model_used"] = data.get("model_used")
                                meta["retrieval_mode"] = data.get("retrieval_mode")
                                meta["latency_ms"] = data.get("latency_ms")
                            
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

                # Stream response using st.write_stream
                st.write_stream(token_stream())

        except Exception as e:
            streaming_success = False

        # Fallback to standard /query if streaming was not reachable
        if not streaming_success:
            try:
                status_box.update(label="🛡️ Checking guardrails & retrieving context...", state="running")
                query_url = f"{BASE_URL}/query"
                q_payload = {"q": user_query, "thread_id": st.session_state.session_id}
                res = requests.post(query_url, json=q_payload, timeout=90)
                data = res.json()

                meta["answer_text"] = data.get("answer", "No response generated.")
                meta["sources"] = data.get("sources", [])
                thought_process = data.get("thought_process", [])
                meta["model_used"] = data.get("model_used")
                meta["retrieval_mode"] = data.get("retrieval_mode")
                meta["latency_ms"] = int((time.perf_counter() - start_time) * 1000)

                # Classify state
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

        # Append assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": meta["answer_text"],
            "sources": meta["sources"],
            "model_used": meta["model_used"],
            "retrieval_mode": meta["retrieval_mode"],
            "thought_process": thought_process,
            "latency_ms": meta["latency_ms"] or int((time.perf_counter() - start_time) * 1000),
            "state_type": meta["state_type"],
            "message_id": meta["message_id"],
            "question_text": user_query
        })
        st.rerun()
