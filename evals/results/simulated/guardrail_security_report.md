# TrustRAG — Guardrails Security Benchmark Evaluation Report

> [!WARNING]
> **SIMULATED — not a classifier result**
> This evaluation was generated in offline simulation mode using fixed benchmark expectations, NOT a live LLM classifier call. Do not cite these numbers as live classifier performance.

- **Dataset**: `evals\data\guardrails_benchmark_v2.json`
- **Evaluation Mode**: `offline (SIMULATED)`
- **Total Benchmark Cases**: `40`

## 1. Executive Summary & Core Security Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | `100.0%` | Proportion of correct binary enforcement decisions. |
| **Recall / TPR** | `100.0%` | Attack/Off-topic detection rate (`TP / (TP + FN)`). |
| **Precision** | `100.0%` | Reliability of block decisions (`TP / (TP + FP)`). |
| **False Negative Rate (FNR)** | `0.0%` | Miss rate where attacks slipped through. |
| **False Positive Rate (FPR)** | `0.0%` | Benign technical inquiries falsely blocked. |
| **Reason Family Alignment** | `100.0%` | Semantic categorization alignment with tolerance. |

## 2. Confusion Matrix

| | Predicted BLOCK | Predicted ALLOW |
| :--- | :---: | :---: |
| **Actual ATTACK / OFF-TOPIC** | **TP: 28** | **FN: 0** |
| **Actual BENIGN / IN-SCOPE** | **FP: 0** | **TN: 12** |

## 3. Per-Category Breakdown

| Category | Cases | Binary Accuracy | Reason Alignment | Evaluation Focus |
| :--- | :---: | :---: | :---: | :--- |
| `direct_jailbreak` | 4 | 100.0% | 100.0% | Explicit persona overrides & DAN attacks |
| `system_prompt_extraction` | 4 | 100.0% | 100.0% | System prompt probing & JSON leaks |
| `roleplay_override` | 4 | 100.0% | 100.0% | Fictional simulations & multi-agent pretexts |
| `obfuscated_encoded` | 4 | 100.0% | 100.0% | Base64, ROT13, Leetspeak, delimiter splitting |
| `quoted_embedded` | 4 | 100.0% | 100.0% | Markdown quotes, code review comments, fake logs |
| `unsafe_request` | 4 | 100.0% | 100.0% | Actionable malware, DDoS scripts, credential cracking |
| `benign_security_terms` | 4 | 100.0% | 100.0% | Defensive hardening (SYN cookies, ASLR, RBAC) |
| `benign_technical` | 4 | 100.0% | 100.0% | Core Kubernetes, Redis, Databricks, OS kernel docs |
| `out_of_scope_benign` | 4 | 100.0% | 100.0% | Jokes, recipes, poetry, sports (Domain boundary) |
| `ambiguous_and_edge_cases` | 4 | 100.0% | 100.0% | Definitions, multi-line YAML manifests, CLI snippets |

## 4. Key Architectural Observations & Security Posture

1. **Binary Enforcement Contract**: The pipeline gating in `app/agents/pipeline.py` strictly evaluates `decision in ('block', 'infra_unavailable')`. The guard fails closed safely without leaking internal prompts.
2. **Defensive vs. Offensive Discrimination**: Defensive cybersecurity questions (e.g. configuring SYN cookies or k8s NetworkPolicies) are accurately permitted, avoiding enterprise operational false alarms.
3. **Reason Family Determinism**: Instruction overrides and encoded payloads exhibit natural semantic overlap between `jailbreak` and `injection`. Tolerating this overlap reflects production reality without compromising security.
4. **Indirect Prompt Injection (IPI) Note**: The query guardrails inspect incoming user prompts (`q`). However, retrieved document context from vector storage is concatenated directly into responder context without secondary inspection. See test suite for controlled canary validation.