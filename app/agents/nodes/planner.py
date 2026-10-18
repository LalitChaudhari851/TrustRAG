import json
import re
import logfire
from typing import Tuple, Optional
from app.config import settings
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

# Pre-router chit-chat patterns (exact / near-exact social or meta-bot queries only)
CHITCHAT_EXACT = {
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
    "thanks", "thank you", "thank you so much", "thanks a lot", "many thanks",
    "bye", "goodbye", "see you", "see ya", "cya",
    "ok", "okay", "sure", "got it", "cool", "understood", "k",
    "who are you", "what are you", "what can you do", "what do you do",
    "who made you", "what is your name", "help"
}

TECH_KEYWORDS = {
    "job", "jobs", "k8s", "kubernetes", "redis", "hpa", "vpa", "cli",
    "pod", "pods", "cronjob", "cronjobs", "api", "rest", "databricks",
    "cluster", "kubectl", "docker", "python", "yaml", "config", "node",
    "container", "autoscale", "autoscaling", "daemon", "service", "ingress",
    "queue", "rpush", "lpop", "rpop", "worker"
}


def is_pure_chitchat(text: str) -> bool:
    """
    Rule-based deterministic pre-router.
    Returns True ONLY if the text is an exact/near-exact social chitchat or meta-conversation query.
    All technical, infrastructure, code, or command questions return False.
    """
    if not text:
        return True
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in CHITCHAT_EXACT:
        return True
    words = cleaned.split()
    if len(words) <= 3 and any(w in {"hi", "hello", "hey"} for w in words):
        if not any(w in TECH_KEYWORDS for w in words):
            return True
    return False


def extract_json_route(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts route ('retrieve' or 'chitchat') and query from LLM text.
    Returns (route, query) or (None, None) if parsing fails.
    """
    if not text:
        return None, None
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end+1].strip()

    try:
        data = json.loads(cleaned)
        route = data.get("route", "").lower().strip()
        query = data.get("query", "").strip() if route == "retrieve" else ""
        if route in ("retrieve", "chitchat"):
            return route, query
    except Exception:
        pass
    return None, None


# Portkey-backed LLM: temperature=0, max_tokens=512 for ample headroom (thinking tokens + JSON)
llm = get_langchain_llm(feature="planner", max_tokens=512)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on conversation history and rules.
    - Rule-based pre-router: only exact/near-exact chit-chat may skip retrieval.
    - Standalone questions: skip LLM planner and retrieve with raw user message.
    - Short follow-ups with history: use LLM rewrite concatenated with original query.
    - Strict JSON output with 1 retry on invalid format.
    - Any error, invalid output, or technical question defaults to RETRIEVE.
    """
    messages = state.get("messages", [])
    history_msgs = []
    history_text = ""
    for msg in messages[:-1]:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        role = "User" if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) == "user" else "Assistant"
        if content:
            history_msgs.append((role, content))
            history_text += f"{role}: {content}\n"

    last_msg = messages[-1] if messages else {}
    user_message = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")

    # 1. Rule-Based Pre-Router Check
    if is_pure_chitchat(user_message):
        logfire.info(f"⚡ Pre-router identified pure chitchat: '{user_message}'")
        raw_output = '{"route": "chitchat"}'
        return {
            "route": "chitchat",
            "current_query": "CONVERSATIONAL",
            "planner_raw_output": raw_output,
            "status": "Handling conversationally (chitchat/memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped", f"Raw: {raw_output}"]
        }

    # 2. Check Query Rewrite Config & Follow-Up Conditions
    rewrite_mode = getattr(settings, "QUERY_REWRITE", "followup_only").lower()
    has_history = len(history_msgs) > 0
    tokens = user_message.lower().split()
    pronouns = {"it", "that", "this", "these", "those", "its", "they", "them"}
    has_pronoun = bool(set(tokens) & pronouns)
    is_short = len(tokens) < 6
    is_followup = has_history and (has_pronoun or is_short)

    if rewrite_mode == "off":
        should_rewrite = False
    elif rewrite_mode == "always":
        should_rewrite = True
    else:  # followup_only (default)
        should_rewrite = is_followup

    # 3. For Standalone Questions (or when rewrite is skipped):
    if not should_rewrite:
        logfire.info(f"⚡ Standalone/direct query (rewrite skipped, mode={rewrite_mode}) - using raw message: '{user_message}'")
        raw_output = f"SKIPPED_REWRITE: raw_query='{user_message}'"
        return {
            "route": "retrieve",
            "current_query": user_message,
            "planner_raw_output": raw_output,
            "status": f"Technical research needed. Searching for: {user_message}",
            "plan": ["Intent: Technical", f"Search Term: {user_message}", "Planner LLM: Skipped (raw query)"]
        }

    # 4. LLM Query Rewrite for Follow-Ups with History
    prompt = f"""You are an intelligent Assistant Planner for an Enterprise Technical Knowledge Base.
Analyze the user follow-up message in the context of the conversation history.
Reformulate a concise search query that resolves pronouns and missing context.

CONVERSATION HISTORY:
{history_text}

LATEST MESSAGE:
"{user_message}"

Decision Rules:
1. Return strict JSON ONLY. No markdown fences, no conversational preamble, no trailing text.
2. Formulate a focused keyword search query that resolves context from the history:
   {{"route": "retrieve", "query": "<resolved standalone search query>"}}
3. If the user message is purely social (greetings, thanks, chitchat), return:
   {{"route": "chitchat"}}

JSON Output:"""

    raw_output = ""
    route, query = None, None

    try:
        with logfire.span("🧠 Planner Decision"):
            resp = llm.invoke(prompt)
            raw_output = resp.content.strip()
            route, query = extract_json_route(raw_output)

            # Retry once on invalid JSON
            if route is None:
                logfire.warning(f"⚠️ Planner emitted non-JSON output: '{raw_output[:100]}'. Retrying once...")
                retry_prompt = f"""Your previous response was not valid JSON. Please reply with ONLY valid JSON format:
{{"route": "retrieve", "query": "<optimized search query>"}}
OR
{{"route": "chitchat"}}

User message: "{user_message}"
JSON:"""
                retry_resp = llm.invoke(retry_prompt)
                raw_output = retry_resp.content.strip()
                route, query = extract_json_route(raw_output)

    except Exception as e:
        logfire.error(f"❌ Planner LLM invocation failed: {e}. Defaulting to retrieve with raw query.")
        raw_output = f"EXCEPTION: {str(e)}"
        route, query = "retrieve", user_message

    # Safety Invariant: Technical questions must NEVER skip retrieval
    if not is_pure_chitchat(user_message) and route != "retrieve":
        logfire.warning(f"🛡️ Pre-router override: Non-chitchat query attempted route '{route}'. Forcing 'retrieve'.")
        route = "retrieve"
        query = query or user_message

    if route == "chitchat":
        return {
            "route": "chitchat",
            "current_query": "CONVERSATIONAL",
            "planner_raw_output": raw_output,
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped", f"Raw: {raw_output}"]
        }

    # A rewrite must be concatenated with the original query, never replace it.
    if query and query.strip() and query.strip().lower() != user_message.strip().lower():
        combined_query = f"{user_message} {query.strip()}"
    else:
        combined_query = user_message

    logfire.info(f"Original query: '{user_message}' | LLM rewritten query: '{query}' | Combined search query: '{combined_query}'")

    return {
        "route": "retrieve",
        "current_query": combined_query,
        "planner_raw_output": raw_output,
        "status": f"Technical research needed. Searching for: {combined_query}",
        "plan": ["Intent: Technical", f"Search Term: {combined_query}", f"Raw: {raw_output}"]
    }
