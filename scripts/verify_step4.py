if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.retrieval.qdrant_service import client, search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

OUTPUT_FILE = "evals/results/verification_4.txt"

# Golden 3 misses and Hard 3 misses
TARGET_QUERIES = [
    {
        "dataset": "Golden 15",
        "id": 4,
        "question": "What is the difference between HPA and VPA in Kubernetes?",
        "expected_source": "pods_autoscale.html",
        "expected_section": "HPA vs. VPA: Which One to Use?",
        "dataset_file": "evals/golden_dataset.json",
        "dataset_commit": "7c23502 (initial) / 24a2b70 (phase 2 schema)"
    },
    {
        "dataset": "Golden 15",
        "id": 7,
        "question": "What Databricks CLI command retrieves the details of a specific job?",
        "expected_source": "job_management.html",
        "expected_section": "Example: Retrieve a job using the CLI",
        "dataset_file": "evals/golden_dataset.json",
        "dataset_commit": "7c23502 (initial) / 24a2b70 (phase 2 schema)"
    },
    {
        "dataset": "Golden 15",
        "id": 11,
        "question": "What are the completions and parallelism fields in a Kubernetes Job spec?",
        "expected_source": "cronjobs.docx",
        "expected_section": "Completions and Parallelism",
        "dataset_file": "evals/golden_dataset.json",
        "dataset_commit": "7c23502 (initial) / 24a2b70 (phase 2 schema)"
    },
    {
        "dataset": "Hard 22",
        "id": 3,
        "question": "What happens when a Kubernetes Job specifies completionMode: Indexed?",
        "expected_source": "cronjobs.docx",
        "expected_section": "Indexed Mode",
        "dataset_file": "evals/data/hard_retrieval_22.json",
        "dataset_commit": "24a2b70 (phase 2 commit)"
    },
    {
        "dataset": "Hard 22",
        "id": 4,
        "question": "In Kubernetes Jobs, what is the default behavior when completionMode is NonIndexed?",
        "expected_source": "cronjobs.docx",
        "expected_section": "NonIndexed (Default)",
        "dataset_file": "evals/data/hard_retrieval_22.json",
        "dataset_commit": "24a2b70 (phase 2 commit)"
    },
    {
        "dataset": "Hard 22",
        "id": 5,
        "question": "What concurrencyPolicy setting prevents concurrent runs of a scheduled CronJob?",
        "expected_source": "cronjobs.docx",
        "expected_section": "Schedule Configuration",
        "dataset_file": "evals/data/hard_retrieval_22.json",
        "dataset_commit": "24a2b70 (phase 2 commit)"
    }
]

lines = []
lines.append("================================================================================")
lines.append("VERIFICATION ITEM 4: 6 SECTION MISSES (EXPECTED VS RETRIEVED TOP-5 CHUNKS)")
lines.append("================================================================================")

for idx, item in enumerate(TARGET_QUERIES, 1):
    q = item["question"]
    lines.append(f"\n[{idx}/6] Dataset: {item['dataset']} | Query ID: {item['id']}")
    lines.append(f"  Question:         {q}")
    lines.append(f"  Dataset File:     {item['dataset_file']}")
    lines.append(f"  Origin Commit:    {item['dataset_commit']}")
    lines.append(f"  Expected Source:  {item['expected_source']}")
    lines.append(f"  Expected Section: {item['expected_section']}")
    lines.append("  ------------------------------------------------------------------------------")
    lines.append("  Retrieved Top-5 Chunks (Full V2 Hybrid + FlashRank):")

    retrieved = search_enterprise_knowledge(q, limit=15, hybrid=True)
    reranked = rerank_documents(q, retrieved, top_n=5, min_score=0.0, relative_margin=1.0)

    for rank, doc in enumerate(reranked, 1):
        src = doc.get("source", "Unknown")
        sec = doc.get("section") or "[NONE]"
        content = (doc.get("content") or doc.get("text") or "").replace("\n", " ").strip()
        preview_150 = content[:150]
        score = doc.get("rerank_score", doc.get("score", 0.0))
        lines.append(f"    Rank {rank}:")
        lines.append(f"      Source:  {src}")
        lines.append(f"      Section: {sec}")
        lines.append(f"      Score:   {score:.4f}" if isinstance(score, float) else f"      Score:   {score}")
        lines.append(f"      Preview: {preview_150}...")

lines.append("\n" + "="*80)
lines.append("EXPLANATION: WHY EVERY SECTION WAS 'General' IN EARLIER BASELINE REPORT")
lines.append("="*80)
lines.append("1. Root Cause in Legacy Ingestion (Phase 1):")
lines.append("   - In Phase 1 (app/ingestion/processor.py), documents were chunked and uploaded to the")
lines.append("     original 'enterprise_rag' Qdrant collection with payload schema:")
lines.append("       {\"source\": filename, \"source_type\": ..., \"chunk_index\": ..., \"text\": ...}")
lines.append("   - Crucially, the Phase 1 loaders did NOT parse or attach the 'section' metadata to the payload.")
lines.append("   - As a result, the 'section' key did NOT exist in any point payload in the legacy collection.")
lines.append("")
lines.append("2. Fallback Mechanism in scripts/run_baseline_retrieval.py:")
lines.append("   - When evaluating baseline retrieval against the legacy collection, line 73 of")
lines.append("     scripts/run_baseline_retrieval.py executed:")
lines.append("       'section': p.payload.get('section', 'General')")
lines.append("   - Because 'section' was absent from every legacy payload, Python defaulted 100% of the")
lines.append("     retrieved points to the literal string 'General'.")
lines.append("   - Consequently, in baseline_retrieval.json, every single retrieved context was labeled 'General',")
lines.append("     even though the source documents (PDF, DOCX, HTML, PPTX) contained structured section headers.")
lines.append("   - This was resolved in Phase 2 by implementing structured section parsing across all document loaders.")

lines.append("\n" + "="*80)
lines.append("GOLDEN QUERIES WHOSE KEY PHRASE IS ABSENT FROM TOP-15 (PRE-RERANK)")
lines.append("="*80)
lines.append("In retrieval_ab_full.json ('golden_full_v2' -> 'v2_full_hybrid'), exactly 3 queries have key_phrase_hit_15 = False:")
lines.append("1. Golden Q3:")
lines.append("   - Question:   'How do you add items to the work queue using Redis CLI?'")
lines.append("   - Key Phrase: 'rpush job2'")
lines.append("   - Reason:     While chunks describing the Redis work queue are retrieved, the exact command literal")
lines.append("                 'rpush job2' is split across sub-steps and was ranked at position > 15.")
lines.append("2. Golden Q10:")
lines.append("   - Question:   'When was the document regarding running automated jobs with Kubernetes CronJobs created?'")
lines.append("   - Key Phrase: '2023-05-10'")
lines.append("   - Reason:     The date '2023-05-10' resides in the document creation header/metadata of cronjobs.docx")
lines.append("                 and was not ranked in top-15 by semantic query embedding.")
lines.append("3. Golden Q15:")
lines.append("   - Question:   'What command verifies that the Minikube cluster is running properly?'")
lines.append("   - Key Phrase: 'kubectl cluster-info'")
lines.append("   - Reason:     Top retrieved passages discuss 'minikube status' and 'kubectl get nodes' rather than")
lines.append("                 'kubectl cluster-info'.")
lines.append("")
lines.append("WHERE GOLDEN KEY PHRASES ARE DEFINED:")
lines.append("   - Golden key phrases are defined in 'scripts/run_retrieval_ab.py' in the dictionary:")
lines.append("       GOLDEN_KEY_TERMS = { 1: 'redis-pod.yaml', ..., 3: 'rpush job2', 10: '2023-05-10', 15: 'kubectl cluster-info' }")
lines.append("   - They are NOT present in 'evals/golden_dataset.json' (which only defines id, question, expected_source,")
lines.append("     expected_section, reference, and relevant_contexts). In contrast, 'evals/data/hard_retrieval_22.json'")
lines.append("     has 'key_term' explicitly defined inside each JSON object.")

full_content = "\n".join(lines) + "\n"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_content)

print(full_content)
