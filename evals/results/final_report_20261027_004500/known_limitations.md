# TrustRAG — Known Limitations & Technical Boundaries

**Report Directory**: `evals/results/final_report_20261027_004500/`  
**Document Status**: Rigorous & Honest Technical Disclosure  
**Audience**: Technical Reviewers, Hiring Managers, and Systems Architects  

---

## 1. Evaluation Benchmark & Annotation Boundaries

1. **Disputed Batch 1 Annotations (`seed_001` to `seed_005`)**:
   - In earlier development runs, chunk UUIDs for `seed_001` through `seed_005` were recorded against an older corpus index snapshot. When the collection was re-indexed with small-fragment merging (`v3.1-tokenguard-merge-384max`), the generated point UUIDs shifted.
   - Although the model retrieves valid, highly factual context and answers all 5 questions correctly (100% pass rate), the historical candidate UUID mapping remains disputed.
   - **Resolution Policy**: `seed_001`–`seed_005` are segregated into their own cohort and excluded from clean benchmark claims.

2. **Synthetic Benchmark Nature & Sample Size Power ($N=50$)**:
   - The evaluation benchmark consists of 50 carefully crafted questions across Kubernetes, Redis, and Databricks technical domains.
   - While effective for regression testing and comparative ablations, a dev sample size of $N=50$ does not constitute a statistically conclusive enterprise benchmark. Confidence intervals on a 50-query sample remain broad ($\pm 10–14\%$).
   - Claims of superiority must be understood as directional engineering findings rather than definitive population-level claims.

3. **Absence of Large-Scale Human Annotation Campaign**:
   - Relevance grades and expected answers were established through semi-automated extraction with manual engineering spot-checks.
   - No multi-annotator inter-rater reliability (e.g., Cohen's Kappa or Fleiss' Kappa) has been conducted.

---

## 2. Ingestion & Retrieval Pipeline Boundaries

1. **Ingestion Chunking Disparity Between Target and Background Corpora**:
   - **Technical Target Corpus** (`DATA/true_data`, 6 documents, 197 chunks in Qdrant) was indexed using policy `v3.1-tokenguard-merge-384max`. Text fragments shorter than 15 tokens are merged into preceding section headers, ensuring complete preservation of command syntax (e.g., `redis-cli`, `rpush`).
   - **Noisy Background Corpus** (`DATA/noisy_data`, 58 documents, 5,871 chunks in Qdrant) was indexed using legacy policy `v3.0-tokenguard-drop`, where fragments under 15 tokens were discarded.
   - This difference in chunk boundary density is an intentional design tradeoff to keep background noise indexing fast while prioritizing precision on target technical documents.

2. **No Optical Character Recognition (OCR)**:
   - Ingestion relies entirely on local, CPU-bound python libraries (`pypdf`, `pdfplumber`, `python-docx`, `python-pptx`, `bs4`).
   - Scanned raster image PDFs, flowchart diagrams, and bitmap architecture diagrams cannot be parsed. Text embedded inside diagrams is inaccessible to retrieval.

3. **Candidate Pool Truncation Risk ($K=15$ vs $K=25$)**:
   - Under $K=15$, hybrid retrieval dropped candidate position 17 for `seed_009` (HPA vs VPA), resulting in false abstention. Expanding to $K=25$ resolved this issue with negligible CPU overhead (+35ms in FlashRank).
   - In corpora with hundreds of thousands of chunks, candidate pool sizing must be dynamically calibrated against vector collection density.

---

## 3. Safety Guardrail & LLM Gateway Nuances

1. **NVIDIA NeMo Guardrails with Instruction Models**:
   - NeMo Guardrails relies on few-shot intent classification where the underlying LLM must emit exact canonical intent tokens (e.g., `attempt jailbreak`).
   - Smaller or highly instruction-aligned models (such as `qwen/qwen3.8-27b`) frequently emit natural language refusal text directly (`"I cannot fulfill that request..."`) instead of the classification token. NeMo interprets this conversational response as an unrecognized intent, which can cause the Colang flow to bypass downstream protection.
   - **Production Remedy**: `GUARD_MODE=single_call` is maintained as the primary production default. It uses a structured Pydantic JSON schema with automatic repair and strict fail-closed safety, completely immunizing the pipeline against conversational bypass.

2. **Free-Tier Provider Quota & Rate-Limit Dependencies**:
   - The demo environment connects to Groq API endpoints (`qwen/qwen3.8-27b` and `openai/gpt-oss-20b`).
   - Free/developer tier rate limits enforce requests-per-minute (RPM) and tokens-per-minute (TPM) ceilings. Under burst concurrency, queries may experience backoff retries managed by Tenacity and the Portkey gateway.

---

## 4. Operational & Deployment Scope

1. **Streamlit Demo Interface vs Enterprise Frontend**:
   - The included Streamlit UI (`ui/app.py`) is an interactive operational prototype designed for real-time demonstration, developer inspection, and trace auditing.
   - For a production deployment, Streamlit should be replaced with an enterprise React/Next.js application integrated with OpenID Connect (OIDC) / SAML Single Sign-On, edge CDN caching, and centralized role-based access control (RBAC).

2. **Client-Supplied Thread Identifiers**:
   - In `/query`, the `thread_id` parameter is client-supplied. While isolated via SQLite/Postgres checkpointer keys, production security requires thread IDs to be cryptographically bound to authenticated user JWT claims to prevent unauthorized cross-session access.
