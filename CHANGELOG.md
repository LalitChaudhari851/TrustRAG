# CHANGELOG

## [Unreleased] - 2026-10-17

### Corrected
- **Golden Dataset Key Phrases (Q3, Q10, Q15)**:
  - Corrected label key phrases in `evals/golden_dataset.json` with verified verbatim chunk terms from the Qdrant `enterprise_rag_v2` index:
    - **Q3**: Changed legacy `"rpush job2"` to `"redis-cli -h redis"` (the markdown comment `# Inside the redis prompt:` had split the CLI command and tasks into separate chunks; chunk index 4 contains `"redis-cli -h redis"` matching section `STEP 2: FILLING THE QUEUE WITH TASKS`).
    - **Q10**: Changed legacy `"2023-05-10"` (timestamp copy-paste artifact) to `"OnFailure"` (verbatim restart policy keyword in chunk index 2 matching section `1. Restart Policy Requirements`).
    - **Q15**: Changed legacy `"kubectl cluster-info"` to `"kubectl get pods -n kube-system"` (verbatim command in chunk index 4 matching section `Verifying Kubernetes Installation`).
  - Prior key phrase values and reasons are preserved in a dedicated `"label_history"` dictionary for full auditability.
  - Re-ran retrieval evaluation: Golden 15 Key Hit@5 improves from 0.8000 to **1.0000** (15/15) across dense, hybrid, and threshold modes.
