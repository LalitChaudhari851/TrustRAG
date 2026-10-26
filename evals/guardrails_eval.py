"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""


import time
import copy
import requests
import logfire

API_URL = "http://localhost:8000/query"


def _is_blocked(response_json: dict) -> bool:
    tp = response_json.get("thought_process") or []
    return any("guardrails fired" in step.lower() for step in tp)


def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked and result (TP/TN/FP/FN) to each sample in place.
    Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, n, sample["input"])

            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                infra_error = False
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": sample["input"], "thread_id": f"guardrail_eval_{i}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    blocked = _is_blocked(resp.json())

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    blocked = False
                    infra_error = True

                except Exception as e:
                    logfire.error(f"❌ Guardrails test error: {e}")
                    blocked = False
                    infra_error = True

                expected = sample["expected_blocked"]
                sample["actual_blocked"] = None if infra_error else blocked

                if infra_error:
                    sample["result"] = "INFRA_ERROR"
                elif expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                logfire.info(
                    f"🛡️ {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=sample["actual_blocked"],
                    input_preview=sample["input"][:60],
                )

            time.sleep(2)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    valid_results = [r for r in results if r.get("result") in ("TP", "TN", "FP", "FN")]
    infra_errors = sum(1 for r in results if r.get("result") == "INFRA_ERROR")

    tp = sum(1 for r in valid_results if r["result"] == "TP")
    tn = sum(1 for r in valid_results if r["result"] == "TN")
    fp = sum(1 for r in valid_results if r["result"] == "FP")
    fn = sum(1 for r in valid_results if r["result"] == "FN")

    valid_count = len(valid_results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / valid_count if valid_count > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "valid_total": valid_count,
        "infra_errors": infra_errors,
        "correct": tp + tn,
    }


