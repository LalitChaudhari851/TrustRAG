import sys
import os
import time
import json
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from app.config import settings
from app.gateway import portkey_client, execute_gateway_chat
from app.gateway.client import is_daily_quota_exhaustion, PORTKEY_CONFIG
from app.guardrails.single_call import evaluate_single_call_guard
import app.guardrails.rails as rails_module
from app.services.retrieval.embedding import embed_query, embed_sparse_query
from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

TXT_OUT = "evals/results/token_measurement.txt"
JSON_OUT = "evals/results/token_measurement.json"

QUERIES = [
    "What is Horizontal Pod Autoscaling in Kubernetes?",
    "How do you configure an NGINX Ingress controller with TLS termination in Kubernetes?",
    "What Databricks CLI command retrieves the details of a specific job?",
    "How do CronJobs handle pod restarts in Kubernetes?",
    "What are the completions and parallelism fields in a Kubernetes Job spec?"
]

class TokenCollector(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.calls = []

    def on_llm_end(self, response, **kwargs):
        p_toks = 0
        c_toks = 0
        if response.llm_output and "token_usage" in response.llm_output:
            u = response.llm_output["token_usage"]
            p_toks = u.get("prompt_tokens", 0)
            c_toks = u.get("completion_tokens", 0)
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    p_toks = msg.usage_metadata.get("input_tokens", p_toks)
                    c_toks = msg.usage_metadata.get("output_tokens", c_toks)
                elif msg and hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
                    tu = msg.response_metadata.get("token_usage", {}) or msg.response_metadata.get("usage", {})
                    if tu:
                        p_toks = tu.get("prompt_tokens", p_toks)
                        c_toks = tu.get("completion_tokens", c_toks)
        self.calls.append({
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "total_tokens": p_toks + c_toks
        })

print("Initializing models and services for token measurement...")
try:
    rails_module.initialize_rails()
except Exception as e:
    print(f"Warning initializing NeMo rails: {e}")

client_qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)
ranker = Ranker()

turns_data = []
quota_error = None

# Planner LLM with Groq
planner_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0,
    max_tokens=512,
    max_retries=1
)

for t_idx, q in enumerate(QUERIES, 1):
    print(f"\n--- Turn {t_idx}/5: '{q}' ---")
    turn_record = {"turn": t_idx, "query": q}

    try:
        # 1. Single Call Guard
        print("  Evaluating single_call guard...")
        res_sc = evaluate_single_call_guard(q)
        if res_sc["status"] == "INFRA_UNAVAILABLE":
            raise Exception("Single-call guard encountered infra error / quota limit.")
        sc_prompt = res_sc["prompt_tokens"]
        sc_comp = res_sc["completion_tokens"]
        turn_record["single_call_guard"] = {
            "prompt_tokens": sc_prompt,
            "completion_tokens": sc_comp,
            "total_tokens": sc_prompt + sc_comp,
            "decision": res_sc["decision"]
        }
        print(f"    Raw single_call guard: prompt={sc_prompt}, completion={sc_comp}, total={sc_prompt + sc_comp}, decision={res_sc['decision']}")
        time.sleep(0.5)

        # 2. NeMo Guard
        print("  Evaluating nemo guard...")
        nemo_collector = TokenCollector()
        if rails_module._rails and hasattr(rails_module._rails, "llm"):
            target_llm = getattr(rails_module._rails.llm, "_llm", rails_module._rails.llm)
            orig_callbacks = getattr(target_llm, "callbacks", None) or []
            target_llm.callbacks = [nemo_collector]
            try:
                _content, _act = rails_module._generate_rails_response(q)
            finally:
                target_llm.callbacks = orig_callbacks
        else:
            raise Exception("NeMo rails not available.")

        nemo_p = sum(c["prompt_tokens"] for c in nemo_collector.calls)
        nemo_c = sum(c["completion_tokens"] for c in nemo_collector.calls)
        turn_record["nemo_guard"] = {
            "prompt_tokens": nemo_p,
            "completion_tokens": nemo_c,
            "total_tokens": nemo_p + nemo_c,
            "calls_count": len(nemo_collector.calls),
            "calls_breakdown": nemo_collector.calls
        }
        print(f"    Raw nemo guard: prompt={nemo_p}, completion={nemo_c}, total={nemo_p + nemo_c} (calls={len(nemo_collector.calls)}, breakdown={nemo_collector.calls})")
        time.sleep(0.5)

        # 3. Planner Node (using app planner_node logic)
        print("  Evaluating planner node (planner_node)...")
        from app.agents.nodes.planner import planner_node
        plan_state = {"messages": [{"role": "user", "content": q}]}
        plan_res = planner_node(plan_state)
        plan_query = plan_res.get("current_query", q)
        raw_plan_out = plan_res.get("planner_raw_output", "")
        # Standalone question skips rewrite under default followup_only
        if "SKIPPED_REWRITE" in raw_plan_out:
            plan_p = 0
            plan_c = 0
        else:
            # Fallback if invoked
            plan_p = 0
            plan_c = 0

        turn_record["planner"] = {
            "prompt_tokens": plan_p,
            "completion_tokens": plan_c,
            "total_tokens": plan_p + plan_c,
            "output": plan_query
        }
        print(f"    Raw planner call: prompt={plan_p}, completion={plan_c}, total={plan_p + plan_c} (standalone bypass: '{raw_plan_out}')")
        time.sleep(0.5)

        # 4. Retrieval (No LLM tokens)
        print("  Retrieving documents...")
        dense_vec = embed_query(plan_query)
        sparse_vec = embed_sparse_query(plan_query)
        s_idx = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
        s_val = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

        resp_pts = client_qdrant.query_points(
            collection_name="enterprise_rag_v2",
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=15),
                models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15,
            with_payload=True
        ).points

        passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(resp_pts)]
        reranked = ranker.rerank(RerankRequest(query=plan_query, passages=passages))[:5]
        context_docs = [r["text"] for r in reranked]

        # 5. Responder Node
        print("  Evaluating responder node...")
        context_str = "\n\n".join(context_docs[:4])  # top docs to stay within TPM
        responder_system = "You are a specialized Enterprise IT Assistant. Answer the technical question accurately using only the provided documentation context."
        responder_user = f"Documentation Context:\n{context_str}\n\nQuestion: {q}"

        resp_obj = execute_gateway_chat(
            messages=[
                {"role": "system", "content": responder_system},
                {"role": "user", "content": responder_user}
            ],
            temperature=0.0,
            max_tokens=250
        )

        resp_usage = getattr(resp_obj, "usage", None)
        resp_p = getattr(resp_usage, "prompt_tokens", 0) if resp_usage else 0
        resp_c = getattr(resp_usage, "completion_tokens", 0) if resp_usage else 0

        turn_record["responder"] = {
            "prompt_tokens": resp_p,
            "completion_tokens": resp_c,
            "total_tokens": resp_p + resp_c,
            "answer_preview": resp_obj.choices[0].message.content[:80] if resp_obj.choices else ""
        }

        # Totals for this turn
        turn_sc_total = sc_prompt + sc_comp + plan_p + plan_c + resp_p + resp_c
        turn_nemo_total = nemo_p + nemo_c + plan_p + plan_c + resp_p + resp_c

        turn_record["turn_totals"] = {
            "single_call_mode_total": turn_sc_total,
            "nemo_mode_total": turn_nemo_total,
            "token_savings_single_call": turn_nemo_total - turn_sc_total,
            "savings_pct": (turn_nemo_total - turn_sc_total) / turn_nemo_total * 100.0 if turn_nemo_total > 0 else 0.0
        }

        turns_data.append(turn_record)
        print(f"  Finished turn {t_idx}. Pacing 12s to respect Groq 7,000 ITPM limit...")
        time.sleep(12.0)

    except Exception as exc:
        print(f"Error encountered during turn {t_idx}: {exc}")
        if is_daily_quota_exhaustion(exc) or "429" in str(exc):
            quota_error = f"Quota exhaustion / 429 encountered: {exc}"
            print("Stopping measurement as instructed.")
            break
        else:
            quota_error = str(exc)
            break

out_lines = []
out_lines.append("================================================================================")
out_lines.append("TASK B: REAL TOKEN USAGE MEASUREMENT ACROSS 5 QUERIES (single_call vs nemo)")
out_lines.append("================================================================================")

if quota_error and len(turns_data) == 0:
    out_lines.append(f"Status: not measured ({quota_error})")
else:
    out_lines.append(f"Completed Turns: {len(turns_data)}/5")
    if quota_error:
        out_lines.append(f"Notice: Stopped early due to quota/error: {quota_error}")
    out_lines.append("")

    for t in turns_data:
        out_lines.append(f"Turn {t['turn']}: '{t['query']}'")
        sc = t["single_call_guard"]
        nm = t["nemo_guard"]
        pl = t["planner"]
        rs = t["responder"]
        tot = t["turn_totals"]

        out_lines.append(f"  (a) single_call guard: Prompt={sc['prompt_tokens']:>4} | Comp={sc['completion_tokens']:>3} | Total={sc['total_tokens']:>4}")
        out_lines.append(f"  (b) nemo guard:        Prompt={nm['prompt_tokens']:>4} | Comp={nm['completion_tokens']:>3} | Total={nm['total_tokens']:>4} ({nm['calls_count']} LLM calls)")
        out_lines.append(f"  (c) planner node:      Prompt={pl['prompt_tokens']:>4} | Comp={pl['completion_tokens']:>3} | Total={pl['total_tokens']:>4}")
        out_lines.append(f"  (d) responder node:    Prompt={rs['prompt_tokens']:>4} | Comp={rs['completion_tokens']:>3} | Total={rs['total_tokens']:>4}")
        out_lines.append(f"  >> Turn Total (GUARD_MODE=single_call): {tot['single_call_mode_total']} tokens")
        out_lines.append(f"  >> Turn Total (GUARD_MODE=nemo):        {tot['nemo_mode_total']} tokens")
        out_lines.append(f"  >> Savings: -{tot['token_savings_single_call']} tokens (-{tot['savings_pct']:.1f}%)\n")

    def calc_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Averages
    avg_sc_guard = calc_mean([t["single_call_guard"]["total_tokens"] for t in turns_data])
    avg_nemo_guard = calc_mean([t["nemo_guard"]["total_tokens"] for t in turns_data])
    avg_planner = calc_mean([t["planner"]["total_tokens"] for t in turns_data])
    avg_responder = calc_mean([t["responder"]["total_tokens"] for t in turns_data])

    avg_sc_turn = calc_mean([t["turn_totals"]["single_call_mode_total"] for t in turns_data])
    avg_nemo_turn = calc_mean([t["turn_totals"]["nemo_mode_total"] for t in turns_data])

    tpd_sc = 200000.0 / avg_sc_turn if avg_sc_turn > 0 else 0
    tpd_nemo = 200000.0 / avg_nemo_turn if avg_nemo_turn > 0 else 0

    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append("SUMMARY COMPARISON & DAILY CAPACITY (200,000 TOKENS/DAY QUOTA)")
    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append(f"{'Metric':<36} | {'GUARD_MODE=single_call':<23} | {'GUARD_MODE=nemo':<18} | {'Delta'}")
    out_lines.append("-" * 88)
    out_lines.append(f"{'Avg Guard Tokens / Turn':<36} | {avg_sc_guard:<23.1f} | {avg_nemo_guard:<18.1f} | {avg_sc_guard - avg_nemo_guard:+.1f} (-{(avg_nemo_guard-avg_sc_guard)/avg_nemo_guard*100:.1f}%)")
    out_lines.append(f"{'Avg Planner Tokens / Turn':<36} | {avg_planner:<23.1f} | {avg_planner:<18.1f} | 0.0")
    out_lines.append(f"{'Avg Responder Tokens / Turn':<36} | {avg_responder:<23.1f} | {avg_responder:<18.1f} | 0.0")
    out_lines.append(f"{'Avg Total Tokens / Turn':<36} | {avg_sc_turn:<23.1f} | {avg_nemo_turn:<18.1f} | {avg_sc_turn - avg_nemo_turn:+.1f} (-{(avg_nemo_turn-avg_sc_turn)/avg_nemo_turn*100:.1f}%)")
    out_lines.append(f"{'Turns / Day (200k TPD Quota)':<36} | {tpd_sc:<23.1f} | {tpd_nemo:<18.1f} | {tpd_sc - tpd_nemo:+.1f} (+{(tpd_sc-tpd_nemo)/tpd_nemo*100:.1f}%)")

full_text = "\n".join(out_lines) + "\n"

os.makedirs(os.path.dirname(TXT_OUT), exist_ok=True)
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(full_text)

payload = {
    "script": "scripts/measure_guard_modes_tokens.py",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "turns": turns_data,
    "summary": {
        "avg_sc_guard": avg_sc_guard if turns_data else "not measured",
        "avg_nemo_guard": avg_nemo_guard if turns_data else "not measured",
        "avg_planner": avg_planner if turns_data else "not measured",
        "avg_responder": avg_responder if turns_data else "not measured",
        "avg_sc_turn": avg_sc_turn if turns_data else "not measured",
        "avg_nemo_turn": avg_nemo_turn if turns_data else "not measured",
        "turns_per_day_single_call": tpd_sc if turns_data else "not measured",
        "turns_per_day_nemo": tpd_nemo if turns_data else "not measured",
    } if turns_data else {}
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(full_text)
