import os
import sys
import time
import json
import statistics
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


RESULTS_FILE = "evals/results/real_query_tokens_breakdown.json"

QUERIES = [
    "What are the restart policies for CronJobs in Kubernetes?",
    "How do I monitor the status of a Kubernetes Job with kubectl?",
    "What is Horizontal Pod Autoscaling and what metrics does it use?",
    "How do you configure SRIOV network virtual functions on Intel NICs?",
    "What does the backoffLimit parameter do in a Kubernetes Job manifest?"
]


def main():
    print("=== STEP 6: MEASURING REAL TOKEN USAGE ACROSS 5 REAL /QUERY TURNS ===")
    
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    turn_measurements = []

    for i, q in enumerate(QUERIES, 1):
        print(f"\n--- Turn {i}/5: '{q}' ---")
        
        # 1. Guardrail check simulation (exact Colang prompt format sent to Groq)
        guard_prompt = (
            "Determine if the user query is asking for off-topic, jailbreak, or harmful content.\n"
            "Domain: Enterprise Kubernetes, Intel hardware, enterprise networking.\n"
            f"User Query: \"{q}\"\n"
            "Classify intent:"
        )
        t0 = time.perf_counter()
        guard_resp = llm.invoke([HumanMessage(content=guard_prompt)])
        lat_guard = (time.perf_counter() - t0) * 1000
        guard_usage = guard_resp.response_metadata.get("token_usage", {})
        print(f"  Guard:     Prompt: {guard_usage.get('prompt_tokens', 0):4d} | Completion: {guard_usage.get('completion_tokens', 0):3d} | Total: {guard_usage.get('total_tokens', 0):4d} ({lat_guard:.1f}ms)")
        
        time.sleep(2)  # Avoid burst rate limits

        # 2. Planner Stage (exact prompt from app/agents/nodes/planner.py)
        planner_prompt = (
            "You are an intelligent Assistant Planner for an Enterprise Knowledge Base.\n"
            "Analyze the conversation history and the latest user message.\n\n"
            "CONVERSATION HISTORY:\n\n"
            f"LATEST MESSAGE:\n\"{q}\"\n\n"
            "Decision Rules:\n"
            "1. Output 'CONVERSATIONAL' ONLY if the message is purely social (greetings, thanks, farewells) or a meta-question about the conversation itself.\n"
            "2. If the message is ANY technical question about Kubernetes, Intel, or Networking, ALWAYS generate an optimized search query to retrieve authoritative documentation.\n\n"
            "Output ONLY 'CONVERSATIONAL' or the search query."
        )
        t0 = time.perf_counter()
        planner_resp = llm.invoke([HumanMessage(content=planner_prompt)])
        lat_planner = (time.perf_counter() - t0) * 1000
        planner_usage = planner_resp.response_metadata.get("token_usage", {})
        print(f"  Planner:   Prompt: {planner_usage.get('prompt_tokens', 0):4d} | Completion: {planner_usage.get('completion_tokens', 0):3d} | Total: {planner_usage.get('total_tokens', 0):4d} ({lat_planner:.1f}ms)")

        time.sleep(2)

        # 3. Retrieval + Reranking to get REAL context chunks
        candidates = search_enterprise_knowledge(q, limit=15, hybrid=True)
        reranked = rerank_documents(q, candidates, top_n=5)
        context_str = "\n\n".join([f"Source: {c.get('source')} (Section: {c.get('section')})\n{c.get('content')}" for c in reranked])

        # 4. Responder Stage (exact prompt from app/agents/nodes/responder.py)
        responder_prompt = (
            "You are a Senior Technical Architect.\n"
            "Answer the question using the TECHNICAL CONTEXT provided.\n\n"
            f"TECHNICAL CONTEXT:\n{context_str}\n\n"
            "CONVERSATION HISTORY:\n\n"
            f"USER QUESTION:\n\"{q}\""
        )
        t0 = time.perf_counter()
        responder_resp = llm.invoke([HumanMessage(content=responder_prompt)])
        lat_responder = (time.perf_counter() - t0) * 1000
        responder_usage = responder_resp.response_metadata.get("token_usage", {})
        print(f"  Responder: Prompt: {responder_usage.get('prompt_tokens', 0):4d} | Completion: {responder_usage.get('completion_tokens', 0):3d} | Total: {responder_usage.get('total_tokens', 0):4d} ({lat_responder:.1f}ms)")

        # Total per turn
        total_prompt = guard_usage.get('prompt_tokens', 0) + planner_usage.get('prompt_tokens', 0) + responder_usage.get('prompt_tokens', 0)
        total_comp = guard_usage.get('completion_tokens', 0) + planner_usage.get('completion_tokens', 0) + responder_usage.get('completion_tokens', 0)
        turn_total = total_prompt + total_comp
        total_lat = lat_guard + lat_planner + lat_responder
        print(f"  => TOTAL TURN {i}: Prompt: {total_prompt} | Completion: {total_comp} | Total: {turn_total} tokens ({total_lat:.1f}ms)")

        turn_measurements.append({
            "turn": i,
            "query": q,
            "guard": {
                "prompt_tokens": guard_usage.get("prompt_tokens", 0),
                "completion_tokens": guard_usage.get("completion_tokens", 0),
                "total_tokens": guard_usage.get("total_tokens", 0),
                "latency_ms": round(lat_guard, 1)
            },
            "planner": {
                "prompt_tokens": planner_usage.get("prompt_tokens", 0),
                "completion_tokens": planner_usage.get("completion_tokens", 0),
                "total_tokens": planner_usage.get("total_tokens", 0),
                "latency_ms": round(lat_planner, 1)
            },
            "responder": {
                "prompt_tokens": responder_usage.get("prompt_tokens", 0),
                "completion_tokens": responder_usage.get("completion_tokens", 0),
                "total_tokens": responder_usage.get("total_tokens", 0),
                "latency_ms": round(lat_responder, 1)
            },
            "turn_totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_comp,
                "total_tokens": turn_total,
                "latency_ms": round(total_lat, 1)
            }
        })

        time.sleep(3)

    # Averages
    avg_guard_prompt = statistics.mean([t["guard"]["prompt_tokens"] for t in turn_measurements])
    avg_guard_comp = statistics.mean([t["guard"]["completion_tokens"] for t in turn_measurements])
    avg_guard_total = statistics.mean([t["guard"]["total_tokens"] for t in turn_measurements])

    avg_planner_prompt = statistics.mean([t["planner"]["prompt_tokens"] for t in turn_measurements])
    avg_planner_comp = statistics.mean([t["planner"]["completion_tokens"] for t in turn_measurements])
    avg_planner_total = statistics.mean([t["planner"]["total_tokens"] for t in turn_measurements])

    avg_responder_prompt = statistics.mean([t["responder"]["prompt_tokens"] for t in turn_measurements])
    avg_responder_comp = statistics.mean([t["responder"]["completion_tokens"] for t in turn_measurements])
    avg_responder_total = statistics.mean([t["responder"]["total_tokens"] for t in turn_measurements])

    avg_turn_prompt = statistics.mean([t["turn_totals"]["prompt_tokens"] for t in turn_measurements])
    avg_turn_comp = statistics.mean([t["turn_totals"]["completion_tokens"] for t in turn_measurements])
    avg_turn_total = statistics.mean([t["turn_totals"]["total_tokens"] for t in turn_measurements])

    daily_limit = 200000
    estimated_turns_per_day = int(daily_limit / avg_turn_total) if avg_turn_total > 0 else 0

    print(f"\n==================================================")
    print(f"AVERAGE TOKEN USAGE PER TURN (REAL PROMPTS)")
    print(f"==================================================")
    print(f"Guard Stage:     Prompt: {avg_guard_prompt:6.1f} | Completion: {avg_guard_comp:5.1f} | Total: {avg_guard_total:6.1f}")
    print(f"Planner Stage:   Prompt: {avg_planner_prompt:6.1f} | Completion: {avg_planner_comp:5.1f} | Total: {avg_planner_total:6.1f}")
    print(f"Responder Stage: Prompt: {avg_responder_prompt:6.1f} | Completion: {avg_responder_comp:5.1f} | Total: {avg_responder_total:6.1f}")
    print(f"--------------------------------------------------")
    print(f"Total Per Turn:  Prompt: {avg_turn_prompt:6.1f} | Completion: {avg_turn_comp:5.1f} | Total: {avg_turn_total:6.1f}")
    print(f"Estimated turns that fit in 200k/day limit: {estimated_turns_per_day} turns/day")

    summary = {
        "script": "scripts/measure_real_query_tokens.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.GROQ_MODEL,
        "daily_limit_tokens": daily_limit,
        "estimated_turns_per_day": estimated_turns_per_day,
        "averages": {
            "guard": {"prompt": round(avg_guard_prompt, 1), "completion": round(avg_guard_comp, 1), "total": round(avg_guard_total, 1)},
            "planner": {"prompt": round(avg_planner_prompt, 1), "completion": round(avg_planner_comp, 1), "total": round(avg_planner_total, 1)},
            "responder": {"prompt": round(avg_responder_prompt, 1), "completion": round(avg_responder_comp, 1), "total": round(avg_responder_total, 1)},
            "turn_totals": {"prompt": round(avg_turn_prompt, 1), "completion": round(avg_turn_comp, 1), "total": round(avg_turn_total, 1)}
        },
        "turns": turn_measurements
    }

    def json_serializer(o):
        if hasattr(o, "__float__"):
            return float(o)
        if hasattr(o, "__int__"):
            return int(o)
        return str(o)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_serializer)

    print(f"\nSaved real token breakdown report to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    main()
