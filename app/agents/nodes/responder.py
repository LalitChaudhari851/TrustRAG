import time
import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway import portkey_client, extract_cache_status
from app.gateway.client import PORTKEY_CONFIG


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        max_tokens = 200 if query == "CONVERSATIONAL" else 400
        create_params = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        try:
            from app.gateway import execute_gateway_chat
            response = execute_gateway_chat(**create_params)

            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            err_msg = str(e).lower()
            logfire.error(f"LLM Generation failed after retries: {e}")
            if "429" in err_msg or "rate limit" in err_msg:
                fallback_reply = "I apologize, but our reasoning service is currently experiencing very high demand (rate limit reached). Please wait a few moments and try again."
            else:
                fallback_reply = "I apologize, but the reasoning service is temporarily unavailable due to a technical error. Please try again shortly."

            return {
                "final_answer": fallback_reply,
                "status": "error",
                "plan": state["plan"] + ["Error: LLM Generation Failed"],
                "messages": [{"role": "assistant", "content": fallback_reply}]
            }

