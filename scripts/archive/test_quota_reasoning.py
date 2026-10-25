import os
import sys
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def test_model_call(name: str, model: str, messages: list, max_tokens: int = 256, extra_body: dict = None):
    print(f"\n--- Testing: {name} (model={model}) ---")
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
        
    try:
        t0 = time.perf_counter()
        completion = client.chat.completions.create(**kwargs)
        lat = (time.perf_counter() - t0) * 1000
        
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Check for reasoning tokens
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
        
        content = completion.choices[0].message.content
        has_think_tag = "<think>" in (content or "")
        
        print(f"  Latency:            {lat:.1f}ms")
        print(f"  Prompt tokens:      {prompt_tokens}")
        print(f"  Completion tokens:  {completion_tokens}")
        print(f"  Reasoning tokens:   {reasoning_tokens}")
        print(f"  Total tokens:       {total_tokens}")
        print(f"  Contains <think>:   {has_think_tag}")
        print(f"  Content snippet:    {repr(content[:80]) if content else 'None'}")
        
        return {
            "name": name,
            "model": model,
            "latency_ms": round(lat, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "contains_think_tag": has_think_tag,
            "extra_body": extra_body
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"name": name, "model": model, "error": str(e)}

def main():
    print("=== MEASURING TOKENS AND REASONING CAPABILITIES ON GROQ ===")
    
    # 1. Test Planner Call (qwen/qwen3.8-27b default vs hidden reasoning)
    planner_prompt = [
        {"role": "system", "content": "You are a query planner. Determine whether to retrieve or respond directly. Output JSON."},
        {"role": "user", "content": "How do CronJobs handle pod restarts in Kubernetes?"}
    ]
    r1 = test_model_call("Planner (Default Qwen)", "qwen/qwen3.8-27b", planner_prompt, max_tokens=150)
    time.sleep(3)
    r2 = test_model_call("Planner (Qwen + reasoning_format='hidden')", "qwen/qwen3.8-27b", planner_prompt, max_tokens=150, extra_body={"reasoning_format": "hidden"})
    time.sleep(3)
    
    # 2. Test Responder Call
    responder_prompt = [
        {"role": "system", "content": "You are a technical assistant. Answer using the provided context."},
        {"role": "user", "content": "Context: CronJobs create Jobs which manage Pods. RestartPolicy can be OnFailure or Never.\nQuestion: How do CronJobs handle pod restarts?"}
    ]
    r3 = test_model_call("Responder (Default Qwen)", "qwen/qwen3.8-27b", responder_prompt, max_tokens=256)
    time.sleep(3)
    r4 = test_model_call("Responder (Qwen + reasoning_format='hidden')", "qwen/qwen3.8-27b", responder_prompt, max_tokens=256, extra_body={"reasoning_format": "hidden"})
    time.sleep(3)
    
    # 3. Test Guard Call
    guard_prompt = [
        {"role": "system", "content": "You are a safety guardrail. Classify if the query is off-topic, jailbreak, or technical."},
        {"role": "user", "content": "How do CronJobs handle pod restarts in Kubernetes?"}
    ]
    r5 = test_model_call("Guard (Default Qwen)", "qwen/qwen3.8-27b", guard_prompt, max_tokens=64)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calls": [r1, r2, r3, r4, r5]
    }
    with open("evals/results/quota_reasoning_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved quota reasoning analysis to evals/results/quota_reasoning_analysis.json")

if __name__ == "__main__":
    main()
