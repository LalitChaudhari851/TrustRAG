import json

with open("evals/results/retrieval_audit_experiments.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"{'Ablation':<22} | {'Recall@5':<8} | {'HitRate@5':<9} | {'MRR@5':<8} | {'NDCG@5':<8} | {'RetCount':<8} | {'MH Conjunction':<14} | {'Total ms':<8}")
print("-" * 105)
for k, v in d["ablations"].items():
    s = v["summary"]
    lat = v["latencies_avg_ms"]
    rm = s["relevance_macro_averages"]
    mh = s.get("multi_hop_averages", {})
    r5 = rm["recall@5"]["mean"]
    hr5 = rm["hit_rate@5"]["mean"]
    mrr5 = rm["mrr@5"]["mean"]
    ndcg5 = rm["ndcg@5"]["mean"]
    mean_ret = sum(q["retrieved_count"] for q in v["query_evals"]) / len(v["query_evals"])
    mh_succ = mh.get("multi_hop_success@5", {}).get("mean", 0.0)
    tot_ms = lat["total_pipeline_ms"]
    print(f"{k:<22} | {r5:<8.4f} | {hr5:<9.4f} | {mrr5:<8.4f} | {ndcg5:<8.4f} | {mean_ret:<8.2f} | {mh_succ:<14.4f} | {tot_ms:<8.1f}")

print("\n--- Multi-Hop Query Breakdown across Ablations ---")
for ab in ["current_production", "no_thresholds", "wider_margin_40", "wider_margin_65", "pool30_margin_40", "pool30_margin_65"]:
    print(f"\n[{ab}]")
    for q in d["ablations"][ab]["query_evals"]:
        if q["query_type"] == "answerable_multi_hop":
            qid = q["query_id"]
            cnt = q["retrieved_count"]
            top_s = q.get("top_score")
            cov = q.get("evidence_coverage@5")
            succ = q.get("multi_hop_success@5")
            ts_str = f"{top_s:.3f}" if top_s is not None else "0.000"
            print(f"  {qid}: ret_cnt={cnt}, top_score={ts_str}, cov@5={cov}, mh_succ@5={succ}")
