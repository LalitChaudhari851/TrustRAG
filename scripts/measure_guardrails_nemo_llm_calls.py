import os
import sys
import time
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from nemoguardrails.rails.llm.options import GenerationOptions
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from app.config import settings
import app.guardrails.rails as rails_module
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

RESULTS_FILE = "evals/results/guardrails_nemo_token_usage.json"

QUERIES = [
    "What are the restart policies for CronJobs in Kubernetes?",
    "How do I monitor the status of a Kubernetes Job with kubectl?",
    "What is Horizontal Pod Autoscaling and what metrics does it use?",
    "How do you configure SRIOV network virtual functions on Intel NICs?",
    "What does the backoffLimit parameter do in a Kubernetes Job manifest?"
]


def main():
    print("=== MEASURING NEMO LLM_CALLS AND REAL TURN TOKEN USAGE ===")
    
    # 1. Initialize Rails
    rails_module.initialize_rails()
    rails = rails_module._rails
    
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    measurements = []

    for i, q in enumerate(QUERIES, 1):
        print(f"\n--- Turn {i}/5: '{q}' ---")
        
        # 1. Guard check via NeMo LLMRails with llm_calls logging
        options = GenerationOptions(output_vars=True, log={"activated_rails": True, "llm_calls": True})
        t0 = time.perf_counter()
        rail_res = rails.generate(messages=[{"role": "user", "content": q}], options=options)
        guard_lat = (time.perf_counter() - t0) * 1000
        
        llm_calls_log = getattr(rail_res.log, "llm_calls", []) if hasattr(rail_res, "log") else []
        guard_calls_count = len(llm_calls_log)
        
        guard_prompt_tokens = sum(getattr(c, "prompt_tokens", 0) for c in llm_calls_log)
        guard_comp_tokens = sum(getattr(c, "completion_tokens", 0) for c in llm_calls_log)
        guard_total_tokens = sum(getattr(c, "total_tokens", 0) for c in llm_calls_log)

        # Fallback if provider metadata didn't populate total_tokens on call objects
        if guard_total_tokens == 0 and llm_calls_log:
            # Approximate from character count
            for c in llm_calls_log:
                p_tok = int(len(getattr(c, "prompt", "")) / 4)
                c_tok = int(len(getattr(c, "completion", "")) / 4)
                guard_prompt_tokens += p_tok
                guard_comp_tokens += c_tok
                guard_total_tokens += (p_tok + c_tok)

        print(f"  Guard:     Calls: {guard_calls_count} | Prompt: {guard_prompt_tokens} | Comp: {guard_comp_tokens} | Total: {guard_total_tokens} ({guard_lat:.1f}ms)")
        for idx, c in enumerate(llm_calls_log, 1):
            print(f"    Call {idx}: task='{getattr(c, 'task', 'unknown')}' | p_tokens={getattr(c, 'prompt_tokens', 0)} | c_tokens={getattr(c, 'completion_tokens', 0)}")

        time.sleep(2)

        # 2. Planner stage (exact app prompt)
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
        p_prompt = planner_usage.get("prompt_tokens", 0)
        p_comp = planner_usage.get("completion_tokens", 0)
        p_total = planner_usage.get("total_tokens", 0)
        print(f"  Planner:   Prompt: {p_prompt} | Comp: {p_comp} | Total: {p_total} ({lat_planner:.1f}ms)")

        time.sleep(2)

        # 3. Retrieval + Reranking (using calibrated threshold)
        candidates = search_enterprise_knowledge(q, limit=15, hybrid=True)
        reranked = rerank_documents(q, candidates, top_n=5)
        context_str = "\n\n".join([f"Source: {c.get('source')} (Section: {c.get('section')})\n{c.get('content')}" for c in reranked])

        # 4. Responder stage (exact app prompt)
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
        resp_usage = responder_resp.response_metadata.get("token_usage", {})
        r_prompt = resp_usage.get("prompt_tokens", 0)
        r_comp = resp_usage.get("completion_tokens", 0)
        r_total = resp_usage.get("total_tokens", 0)
        print(f"  Responder: Prompt: {r_prompt} | Comp: {r_comp} | Total: {r_total} ({lat_responder:.1f}ms)")

        turn_prompt = guard_prompt_tokens + p_prompt + r_prompt
        turn_comp = guard_comp_tokens + p_comp + r_comp
        turn_total = turn_prompt + turn_comp
        turn_lat = guard_lat + lat_planner + lat_responder
        print(f"  => TOTAL TURN {i}: Prompt: {turn_prompt} | Comp: {turn_comp} | Total: {turn_total} tokens ({turn_lat:.1f}ms)")

        measurements.append({
            "turn": i,
            "query": q,
            "guard": {
                "llm_calls_count": guard_calls_count,
                "prompt_tokens": guard_prompt_tokens,
                "completion_tokens": guard_comp_tokens,
                "total_tokens": guard_total_tokens,
                "latency_ms": round(guard_lat, 1),
                "calls_detail": [
                    {
                        "task": getattr(c, "task", "unknown"),
                        "prompt_tokens": getattr(c, "prompt_tokens", 0),
                        "completion_tokens": getattr(c, "completion_tokens", 0),
                        "duration_s": getattr(c, "duration", 0)
                    }
                    for c in llm_calls_log
                ]
            },
            "planner": {
                "prompt_tokens": p_prompt,
                "completion_tokens": p_comp,
                "total_tokens": p_total,
                "latency_ms": round(lat_planner, 1)
            },
            "responder": {
                "prompt_tokens": r_prompt,
                "completion_tokens": r_comp,
                "total_tokens": r_total,
                "latency_ms": round(lat_responder, 1)
            },
            "turn_totals": {
                "prompt_tokens": turn_prompt,
                "completion_tokens": turn_comp,
                "total_tokens": turn_total,
                "total_latency_ms": round(turn_lat, 1)
            }
        })
        time.sleep(2)

    avg_guard_prompt = statistics.mean([m["guard"]["prompt_tokens"] for m in measurements])
    avg_guard_comp = statistics.mean([m["guard"]["completion_tokens"] for m in measurements])
    avg_guard_total = statistics.mean([m["guard"]["total_tokens"] for m in measurements])
    avg_guard_calls = statistics.mean([m["guard"]["llm_calls_count"] for m in measurements])

    avg_plan_prompt = statistics.mean([m["planner"]["prompt_tokens"] for m in measurements])
    avg_plan_comp = statistics.mean([m["planner"]["completion_tokens"] for m in measurements])
    avg_plan_total = statistics.mean([m["planner"]["total_tokens"] for m in measurements])

    avg_resp_prompt = statistics.mean([m["responder"]["prompt_tokens"] for m in measurements])
    avg_resp_comp = statistics.mean([m["responder"]["completion_tokens"] for m in measurements])
    avg_resp_total = statistics.mean([m["responder"]["total_tokens"] for m in measurements])

    avg_turn_prompt = statistics.mean([m["turn_totals"]["prompt_tokens"] for m in measurements])
    avg_turn_comp = statistics.mean([m["turn_totals"]["completion_tokens"] for m in measurements])
    avg_turn_total = statistics.mean([m["turn_totals"]["total_tokens"] for m in measurements])

    daily_limit = 200000
    turns_per_day = int(daily_limit / avg_turn_total) if avg_turn_total > 0 else 0

    print(f"\n==================================================")
    print("AVERAGE TOKEN BREAKDOWN ACROSS 5 REAL TURNS")
    print(f"==================================================")
    print(f"Guard (NeMo):    Calls: {avg_guard_calls:.1f} | Prompt: {avg_guard_prompt:6.1f} | Comp: {avg_guard_comp:5.1f} | Total: {avg_guard_total:6.1f}")
    print(f"Planner:         Prompt: {avg_plan_prompt:6.1f} | Comp: {avg_plan_comp:5.1f} | Total: {avg_plan_total:6.1f}")
    print(f"Responder:       Prompt: {avg_resp_prompt:6.1f} | Comp: {avg_resp_comp:5.1f} | Total: {avg_resp_total:6.1f}")
    print(f"--------------------------------------------------")
    print(f"Total Per Turn:  Prompt: {avg_turn_prompt:6.1f} | Comp: {avg_turn_comp:5.1f} | Total: {avg_turn_total:6.1f}")
    print(f"Recomputed Turns/Day (200k daily limit): {turns_per_day} turns/day")

    summary = {
        "script": "scripts/measure_guardrails_nemo_llm_calls.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.GROQ_MODEL,
        "daily_limit_tokens": daily_limit,
        "turns_per_day": turns_per_day,
        "averages": {
            "guard": {
                "avg_llm_calls": avg_guard_calls,
                "prompt": round(avg_guard_prompt, 1),
                "completion": round(avg_guard_comp, 1),
                "total": round(avg_guard_total, 1)
            },
            "planner": {
                "prompt": round(avg_plan_prompt, 1),
                "completion": round(avg_plan_comp, 1),
                "total": round(avg_plan_total, 1)
            },
            "responder": {
                "prompt": round(avg_resp_prompt, 1),
                "completion": round(avg_resp_comp, 1),
                "total": round(avg_resp_total, 1)
            },
            "turn_totals": {
                "prompt": round(avg_turn_prompt, 1),
                "completion": round(avg_turn_comp, 1),
                "total": round(avg_turn_total, 1)
            }
        },
        "turns": measurements
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

    print(f"\nSaved report to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    main()
