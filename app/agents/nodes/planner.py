from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry with max_retries=0 under unified gateway policy
llm = get_langchain_llm(feature="planner", max_tokens=50)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = f"""
    You are an intelligent Assistant Planner for an Enterprise Knowledge Base.
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Decision Rules:
    1. Output 'CONVERSATIONAL' ONLY if the message is purely social (greetings, thanks, farewells) or a meta-question about the conversation itself (e.g. "what did I ask earlier?", "who are you?").
    2. If the message is ANY technical question about Kubernetes, Intel, or Networking (including asking for explanations, definitions, architecture, or deep-dives into concepts, even if briefly mentioned in previous turns), ALWAYS generate an optimized search query to retrieve authoritative documentation.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
