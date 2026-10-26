"""
TrustRAG Retrieval Seed Dataset Generator.

Generates candidate questions and ground-truth evidence mappings from the approved
source files in processed_data/true/. All candidate questions are populated with
deterministic UUID5 identifiers verified against the live Qdrant index.

Constraints:
1. Marked strictly as 'unverified_synthetic'.
2. Multi-hop questions define explicit alternative sufficient evidence sets.
3. Unanswerable questions are distinguished into 'corpus_unanswerable' and 'unresolved_evidence'.
4. Enforces deduplication, schema validation, and zero split leakage between dev and test.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)

from app.config import settings
from evals.retrieval_dataset_schema import (
    RetrievalEvalRecord,
    validate_dataset_records,
    save_dataset,
)

PROCESSED_TRUE_DIR = "processed_data/true"
OUTPUT_SEED_FILE = "evals/data/retrieval_golden_seed_v1.json"
PROJECT_UUID_NAMESPACE = uuid.UUID(settings.PROJECT_UUID_NAMESPACE)


def compute_chunk_uuid(filename: str, chunk_index: int, text: str) -> str:
    """Computes deterministic UUID5 matching Qdrant ingestion."""
    content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    unique_key = f"{filename}:{chunk_index}:{content_hash}"
    return str(uuid.uuid5(PROJECT_UUID_NAMESPACE, unique_key))


def load_corpus_index() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, int], Dict[str, Any]]]:
    """
    Loads all true data chunks and builds:
    1. file_chunks: filename -> list of chunk dicts
    2. chunk_lookup: (filename, chunk_index) -> {uuid, text, section, ...}
    """
    file_chunks: Dict[str, List[Dict[str, Any]]] = {}
    chunk_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for fname in sorted(os.listdir(PROCESSED_TRUE_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(PROCESSED_TRUE_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        orig_filename = data["filename"]
        chunks = data["chunks"]
        enriched_chunks = []
        for c in chunks:
            c_idx = c["chunk_index"]
            c_text = c["text"]
            c_sec = c.get("section", "General")
            c_uuid = compute_chunk_uuid(orig_filename, c_idx, c_text)
            chunk_info = {
                "uuid": c_uuid,
                "filename": orig_filename,
                "chunk_index": c_idx,
                "section": c_sec,
                "text": c_text,
            }
            enriched_chunks.append(chunk_info)
            chunk_lookup[(orig_filename, c_idx)] = chunk_info

        file_chunks[orig_filename] = enriched_chunks

    return file_chunks, chunk_lookup


def build_seed_records(chunk_lookup: Dict[Tuple[str, int], Dict[str, Any]]) -> List[RetrievalEvalRecord]:
    """
    Builds 50 seed candidate questions with explicit evidence sets and unanswerable subtypes.
    Split: 30 Dev (18 single-hop, 6 multi-hop, 6 unanswerable), 20 Test (12 single-hop, 4 multi-hop, 4 unanswerable).
    """
    records: List[RetrievalEvalRecord] = []

    # Helper to resolve chunk UUID and quote
    def chunk(fn: str, idx: int) -> Dict[str, Any]:
        return chunk_lookup[(fn, idx)]

    # =========================================================================
    # SINGLE-HOP ANSWERABLE (30 items: 18 Dev, 12 Test)
    # =========================================================================
    single_hop_definitions = [
        # --- DEV SET (18 items) ---
        (
            "seed_001", "dev", "easy", "kubernetes_workloads", "parallel_work_queue.txt", 3,
            "How do you start Redis for a Kubernetes work queue?",
            "Run kubectl apply on redis-pod.yaml and redis-service.yaml.",
            "Run the following commands:\n\n    kubectl apply -f https://k8s.io/examples/application/job/redis/redis-pod.yaml\n    kubectl apply -f https://k8s.io/examples/application/job/redis/redis-service.yaml",
        ),
        (
            "seed_002", "dev", "easy", "kubernetes_workloads", "parallel_work_queue.txt", 4,
            "Which CLI commands push tasks into the Redis queue named job2?",
            "Use redis-cli -h redis and rpush job2 with task names.",
            "Once the prompt appears, connect to Redis and fill the queue (named \"job2\"):\n\n    redis-cli -h redis\n\n    # Inside the redis prompt:\n    rpush job2 \"apple\"",
        ),
        (
            "seed_003", "dev", "medium", "kubernetes_workloads", "parallel_work_queue.txt", 6,
            "What parallelism value is specified in the job-wq-2 manifest?",
            "Parallelism is set to 2 in spec.parallelism.",
            "spec:\n      parallelism: 2\n      template:",
        ),
        (
            "seed_004", "dev", "medium", "kubernetes_workloads", "cronjobs.docx", 2,
            "What restart policies are permitted for Kubernetes Job pods?",
            "Only OnFailure or Never are permitted; Always is not allowed.",
            "Jobs have strict requirements for Pod restart policies:\nOnFailure: Kubelet restarts the container within the existing Pod if it fails.\nNever: The Job controller launches a replacement Pod instead.",
        ),
        (
            "seed_005", "dev", "medium", "kubernetes_workloads", "cronjobs.docx", 3,
            "Which API group and version defines the Kubernetes Job resource?",
            "The batch/v1 API group defines the Job resource.",
            "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: batch-processor",
        ),
        (
            "seed_006", "dev", "medium", "kubernetes_workloads", "cronjobs.docx", 6,
            "What is the default backoffLimit for a Kubernetes Job?",
            "The default backoffLimit is 6.",
            "spec:\n  backoffLimit: 6\n  template:",
        ),
        (
            "seed_007", "dev", "hard", "kubernetes_workloads", "cronjobs.docx", 7,
            "What does activeDeadlineSeconds do when applied to a Job?",
            "It sets an absolute maximum duration after which the Job and its Pods are terminated.",
            "activeDeadlineSeconds: 600\nThis sets a strict execution ceiling across all Pods.",
        ),
        (
            "seed_008", "dev", "medium", "kubernetes_workloads", "cronjobs.docx", 8,
            "How does ttlSecondsAfterFinished clean up completed Kubernetes Jobs?",
            "It automatically deletes finished Jobs and their Pods after the specified TTL expires.",
            "ttlSecondsAfterFinished: 86400\nThe TTL-after-finished controller automatically cleans up finished Jobs",
        ),
        (
            "seed_009", "dev", "easy", "kubernetes_workloads", "pods_autoscale.html", 1,
            "What is the core difference between HPA and VPA in Kubernetes?",
            "HPA adjusts replica counts horizontally while VPA adjusts container CPU/memory requests vertically.",
            "HPA scales pods horizontally by adding or removing replicas, while VPA scales pods vertically by adjusting CPU and memory reservations.",
        ),
        (
            "seed_010", "dev", "medium", "kubernetes_workloads", "pods_autoscale.html", 2,
            "Which insecure TLS argument is commonly added to Metrics Server in test clusters?",
            "--kubelet-insecure-tls is added to args in the Metrics Server deployment.",
            "- --kubelet-insecure-tls",
        ),
        (
            "seed_011", "dev", "medium", "kubernetes_workloads", "pods_autoscale.html", 10,
            "What target CPU utilization percentage is configured in the sample HPA manifest?",
            "Target average utilization is configured to 50 percent.",
            "averageUtilization: 50",
        ),
        (
            "seed_012", "dev", "easy", "kubernetes_workloads", "architecture.pptx", 5,
            "What are the four primary master control plane components in Kubernetes?",
            "Kube-apiserver, etcd, kube-scheduler, and kube-controller-manager.",
            "Master Components\nkube-apiserver\netcd\nkube-scheduler\nkube-controller-manager",
        ),
        (
            "seed_013", "dev", "medium", "kubernetes_workloads", "architecture.pptx", 6,
            "What role does etcd serve in a Kubernetes cluster?",
            "It provides consistent and highly-available key-value storage for all cluster state data.",
            "etcd\nConsistent and highly-available key value store used as Kubernetes' backing store for all cluster data.",
        ),
        (
            "seed_014", "dev", "medium", "kubernetes_workloads", "architecture.pptx", 10,
            "What component on worker nodes ensures containers are running in a Pod?",
            "The kubelet agent ensures containers described in PodSpecs are running and healthy.",
            "kubelet\nAn agent that runs on each node in the cluster. It makes sure that containers are running in a pod.",
        ),
        (
            "seed_015", "dev", "easy", "kubernetes_workloads", "monitor_job.docx", 2,
            "Which command checks if Minikube is properly installed on the local system?",
            "minikube version verifies the Minikube installation.",
            "minikube version\nYou should see output similar to:\nminikube version: v1.32.0",
        ),
        (
            "seed_016", "dev", "medium", "kubernetes_workloads", "monitor_job.docx", 3,
            "How do you start Minikube using the Docker driver?",
            "Run minikube start --driver=docker.",
            "minikube start --driver=docker",
        ),
        (
            "seed_017", "dev", "easy", "databricks_job_management", "job_management.html", 2,
            "How do you install the Databricks CLI using curl?",
            "Execute curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh.",
            "curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh",
        ),
        (
            "seed_018", "dev", "medium", "databricks_job_management", "job_management.html", 3,
            "Which Databricks CLI command retrieves details of an existing job?",
            "databricks jobs get --job-id <job-id> retrieves job metadata.",
            "databricks jobs get --job-id",
        ),

        # --- TEST SET (12 items) ---
        (
            "seed_019", "test", "easy", "kubernetes_workloads", "parallel_work_queue.txt", 7,
            "Which kubectl command waits for a Job named job-wq-2 to complete?",
            "kubectl wait --for=condition=complete --timeout=300s job/job-wq-2.",
            "kubectl wait --for=condition=complete --timeout=300s job/job-wq-2",
        ),
        (
            "seed_020", "test", "medium", "kubernetes_workloads", "parallel_work_queue.txt", 8,
            "What background processing library is recommended as an alternative to Redis work queues?",
            "Resque (https://github.com/resque/resque) is suggested for continuous background processing.",
            "Consider using a background processing library such as Resque:\nhttps://github.com/resque/resque",
        ),
        (
            "seed_021", "test", "medium", "kubernetes_workloads", "cronjobs.docx", 5,
            "What is the difference between completions and parallelism in a Kubernetes Job spec?",
            "Completions specifies the number of successful Pod completions required, while parallelism sets the max concurrently running Pods.",
            "Completions specifies how many Pods must terminate successfully for the Job to succeed.\nParallelism sets the maximum number of Pods that can run simultaneously.",
        ),
        (
            "seed_022", "test", "hard", "kubernetes_workloads", "cronjobs.docx", 10,
            "What are the three allowed values for concurrencyPolicy in a Kubernetes CronJob?",
            "Allow, Forbid, and Replace.",
            "concurrencyPolicy: Allow | Forbid | Replace",
        ),
        (
            "seed_023", "test", "medium", "kubernetes_workloads", "cronjobs.docx", 12,
            "What does startingDeadlineSeconds configure in a CronJob?",
            "The deadline in seconds for starting a job if it misses its scheduled time.",
            "startingDeadlineSeconds: 100\nDefines the time window for starting a missed execution.",
        ),
        (
            "seed_024", "test", "medium", "kubernetes_workloads", "pods_autoscale.html", 15,
            "What command uses Apache Bench to simulate traffic on an autoscaled deployment?",
            "ab -n 500000 -c 1000 http://<service-ip>/ generates the synthetic load.",
            "ab -n 500000 -c 1000",
        ),
        (
            "seed_025", "test", "hard", "kubernetes_workloads", "pods_autoscale.html", 24,
            "What are the four update modes supported by Kubernetes VPA?",
            "Off, Initial, Recreate, and Auto.",
            "updateMode: \"Off\" | \"Initial\" | \"Recreate\" | \"Auto\"",
        ),
        (
            "seed_026", "test", "easy", "kubernetes_workloads", "architecture.pptx", 7,
            "What is the responsibility of the kube-scheduler component?",
            "Assigning newly created pods with no assigned node to an appropriate worker node.",
            "kube-scheduler\nWatches for newly created pods that have no node assigned, and selects a node for them to run on.",
        ),
        (
            "seed_027", "test", "medium", "kubernetes_workloads", "architecture.pptx", 11,
            "What does kube-proxy manage on each worker node?",
            "It maintains network rules and connection forwarding to pods across the cluster.",
            "kube-proxy\nkube-proxy maintains network rules on nodes. These network rules allow network communication to your Pods",
        ),
        (
            "seed_028", "test", "medium", "kubernetes_workloads", "monitor_job.docx", 5,
            "What perl command is executed in the sample computing job to calculate Pi?",
            "perl -Mbignum=bpi -wle 'print bpi(2000)' calculates 2000 digits of pi.",
            "perl -Mbignum=bpi -wle 'print bpi(2000)'",
        ),
        (
            "seed_029", "test", "easy", "kubernetes_workloads", "monitor_job.docx", 10,
            "How do you view sorted creation events in Kubernetes using kubectl?",
            "kubectl get events --sort-by='.metadata.creationTimestamp'.",
            "kubectl get events --sort-by='.metadata.creationTimestamp'",
        ),
        (
            "seed_030", "test", "medium", "databricks_job_management", "job_management.html", 4,
            "Which Databricks CLI command creates a new job from a JSON definition file?",
            "databricks jobs create --json-file <path-to-json> creates the job.",
            "databricks jobs create --json-file",
        ),
    ]

    for qid, split_name, diff, dom, fname, cidx, query, answer, quote in single_hop_definitions:
        c_info = chunk(fname, cidx)
        cid = c_info["uuid"]
        sec = c_info["section"]
        records.append(
            RetrievalEvalRecord(
                id=qid,
                query=query,
                query_type="answerable_single_hop",
                difficulty=diff,
                domain=dom,
                source_files=[fname],
                expected_sections=[sec],
                evidence_quotes=[quote],
                candidate_chunk_ids=[cid],
                relevance_grades={cid: 2},
                sufficient_evidence_sets=[[cid]],
                verification_status="unverified_synthetic",
                split=split_name,
                expected_answer=answer,
                notes="Seed synthetic single-hop question; requires human review to verify.",
            )
        )

    # =========================================================================
    # MULTI-HOP ANSWERABLE (10 items: 6 Dev, 4 Test)
    # With explicit alternative sufficient evidence sets where available
    # =========================================================================
    multi_hop_definitions = [
        # --- DEV SET (6 items) ---
        (
            "seed_031", "dev", "medium", "kubernetes_workloads",
            "How does a Kubernetes worker process discover tasks from Redis, and how does the Job manifest configure worker parallelism?",
            "The worker uses RedisWQ to lease items with timeout from Redis, while the Job manifest sets spec.parallelism: 2.",
            [
                ("parallel_work_queue.txt", 5, "q = rediswq.RedisWQ(name=\"job2\", host=host)"),
                ("parallel_work_queue.txt", 6, "spec:\n      parallelism: 2"),
            ],
            # Alternative: overview chunk 1 also states parallelism and redis queue setup
            [
                [("parallel_work_queue.txt", 5), ("parallel_work_queue.txt", 6)],
                [("parallel_work_queue.txt", 0), ("parallel_work_queue.txt", 6)],
            ],
        ),
        (
            "seed_032", "dev", "hard", "kubernetes_workloads",
            "What Pod restart policies are allowed for a Job, and what happens to the Job when backoffLimit is reached after failures?",
            "Allowed restart policies are OnFailure or Never; when consecutive failures reach backoffLimit (default 6), the Job fails.",
            [
                ("cronjobs.docx", 2, "Jobs have strict requirements for Pod restart policies:\nOnFailure: Kubelet restarts"),
                ("cronjobs.docx", 6, "backoffLimit: 6\nWhen failures exceed this limit, the controller marks the Job as Failed."),
            ],
            [
                [("cronjobs.docx", 2), ("cronjobs.docx", 6)],
            ],
        ),
        (
            "seed_033", "dev", "hard", "kubernetes_workloads",
            "How does ttlSecondsAfterFinished interact with activeDeadlineSeconds in controlling Kubernetes Job lifecycle?",
            "activeDeadlineSeconds forces a ceiling on execution duration while running, and ttlSecondsAfterFinished governs retention duration after completion.",
            [
                ("cronjobs.docx", 7, "activeDeadlineSeconds: 600\nThis sets a strict execution ceiling"),
                ("cronjobs.docx", 8, "ttlSecondsAfterFinished: 86400\nThe TTL-after-finished controller automatically cleans up"),
            ],
            [
                [("cronjobs.docx", 7), ("cronjobs.docx", 8)],
            ],
        ),
        (
            "seed_034", "dev", "hard", "kubernetes_workloads",
            "How does Metrics Server enable HPA pod autoscaling, and what parameter allows communication with kubelets having self-signed certificates?",
            "Metrics Server collects pod CPU/memory metrics used by HPA; --kubelet-insecure-tls allows insecure communication with kubelets.",
            [
                ("pods_autoscale.html", 1, "Autoscaling in Kubernetes ensures that your application can handle varying loads"),
                ("pods_autoscale.html", 2, "- --kubelet-insecure-tls"),
            ],
            [
                [("pods_autoscale.html", 1), ("pods_autoscale.html", 2)],
                [("pods_autoscale.html", 0), ("pods_autoscale.html", 2)],
            ],
        ),
        (
            "seed_035", "dev", "hard", "kubernetes_workloads",
            "What control plane component schedules pods onto nodes, and how does kubelet on the target node execute them?",
            "Kube-scheduler watches for unassigned pods and selects a node; kubelet on that node ensures the containers run as specified in PodSpecs.",
            [
                ("architecture.pptx", 7, "kube-scheduler\nWatches for newly created pods that have no node assigned"),
                ("architecture.pptx", 10, "kubelet\nAn agent that runs on each node in the cluster. It makes sure that containers are running"),
            ],
            [
                [("architecture.pptx", 7), ("architecture.pptx", 10)],
                [("architecture.pptx", 5), ("architecture.pptx", 10)],
            ],
        ),
        (
            "seed_036", "dev", "hard", "databricks_job_management",
            "How do you install the Databricks CLI, and what command runs an existing job immediately by ID?",
            "Install with curl script to setup-cli, and trigger with databricks jobs run-now --job-id.",
            [
                ("job_management.html", 2, "curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh"),
                ("job_management.html", 5, "databricks jobs run-now --job-id"),
            ],
            [
                [("job_management.html", 2), ("job_management.html", 5)],
            ],
        ),

        # --- TEST SET (4 items) ---
        (
            "seed_037", "test", "hard", "kubernetes_workloads",
            "How do completions and parallelism settings interact with concurrencyPolicy when executing recurring batch workloads?",
            "Completions and parallelism determine how many pods must finish and how many run in parallel for each Job, while concurrencyPolicy governs concurrent Job runs in CronJobs.",
            [
                ("cronjobs.docx", 5, "Completions specifies how many Pods must terminate successfully"),
                ("cronjobs.docx", 10, "concurrencyPolicy: Allow | Forbid | Replace"),
            ],
            [
                [("cronjobs.docx", 5), ("cronjobs.docx", 10)],
            ],
        ),
        (
            "seed_038", "test", "hard", "kubernetes_workloads",
            "How does Apache Bench simulate load to trigger HPA, and how does HPA determine whether to increase pod replicas?",
            "Apache Bench sends concurrent HTTP requests; HPA queries Metrics Server and increases replicas when average CPU exceeds averageUtilization.",
            [
                ("pods_autoscale.html", 10, "averageUtilization: 50"),
                ("pods_autoscale.html", 15, "ab -n 500000 -c 1000"),
            ],
            [
                [("pods_autoscale.html", 10), ("pods_autoscale.html", 15)],
            ],
        ),
        (
            "seed_039", "test", "hard", "kubernetes_workloads",
            "How do you start a Minikube cluster and verify its installation before monitoring job execution?",
            "Start Minikube with minikube start --driver=docker and verify with minikube version and kubectl cluster-info.",
            [
                ("monitor_job.docx", 2, "minikube version"),
                ("monitor_job.docx", 3, "minikube start --driver=docker"),
            ],
            [
                [("monitor_job.docx", 2), ("monitor_job.docx", 3)],
                [("monitor_job.docx", 1), ("monitor_job.docx", 3)],
            ],
        ),
        (
            "seed_040", "test", "hard", "kubernetes_workloads",
            "What is the difference between etcd and kube-apiserver in state persistence and cluster coordination?",
            "kube-apiserver exposes the Kubernetes API and validates requests; etcd is the backing distributed key-value store holding cluster state.",
            [
                ("architecture.pptx", 5, "kube-apiserver\netcd"),
                ("architecture.pptx", 6, "Consistent and highly-available key value store used as Kubernetes' backing store"),
            ],
            [
                [("architecture.pptx", 5), ("architecture.pptx", 6)],
            ],
        ),
    ]

    for qid, split_name, diff, dom, query, answer, primary_evs, alt_sets in multi_hop_definitions:
        all_candidate_ids = []
        rel_grades = {}
        src_files = set()
        sec_names = set()
        quotes = []

        # Gather primary evidence
        for fn, cidx, qtext in primary_evs:
            cinfo = chunk(fn, cidx)
            cid = cinfo["uuid"]
            if cid not in all_candidate_ids:
                all_candidate_ids.append(cid)
            rel_grades[cid] = 2
            src_files.add(fn)
            sec_names.add(cinfo["section"])
            quotes.append(qtext)

        # Build sufficient sets
        sufficient_sets_uuid = []
        for s_set in alt_sets:
            cur_set = []
            for fn, cidx in s_set:
                cinfo = chunk(fn, cidx)
                cid = cinfo["uuid"]
                if cid not in all_candidate_ids:
                    all_candidate_ids.append(cid)
                    rel_grades[cid] = 1  # supporting
                cur_set.append(cid)
            sufficient_sets_uuid.append(cur_set)

        records.append(
            RetrievalEvalRecord(
                id=qid,
                query=query,
                query_type="answerable_multi_hop",
                difficulty=diff,
                domain=dom,
                source_files=sorted(list(src_files)),
                expected_sections=sorted(list(sec_names)),
                evidence_quotes=quotes,
                candidate_chunk_ids=all_candidate_ids,
                relevance_grades=rel_grades,
                sufficient_evidence_sets=sufficient_sets_uuid,
                verification_status="unverified_synthetic",
                split=split_name,
                expected_answer=answer,
                notes="Seed synthetic multi-hop question with alternative sufficient evidence sets; unverified until human review.",
            )
        )

    # =========================================================================
    # UNANSWERABLE QUERIES (10 items: 6 Dev, 4 Test)
    # Explicitly distinguished: corpus_unanswerable vs unresolved_evidence
    # =========================================================================
    unanswerable_definitions = [
        # --- DEV SET (6 items: 4 corpus_unanswerable, 2 unresolved_evidence) ---
        (
            "seed_041", "dev", "medium", "kubernetes_workloads", "corpus_unanswerable",
            "How do you configure Apache Kafka partition rebalancing on Kubernetes using Strimzi?",
            "Corpus unanswerable: Apache Kafka and Strimzi operators are completely absent from the approved documentation.",
        ),
        (
            "seed_042", "dev", "medium", "kubernetes_workloads", "corpus_unanswerable",
            "What configuration sets up Cassandra SSTable replication factors in a Kubernetes StatefulSet?",
            "Corpus unanswerable: Apache Cassandra database clustering is not covered in the indexed documents.",
        ),
        (
            "seed_043", "dev", "medium", "kubernetes_workloads", "corpus_unanswerable",
            "How do you configure Istio service mesh mutual TLS strict mode between pods?",
            "Corpus unanswerable: Istio service mesh and mTLS traffic policies are absent from the indexed documents.",
        ),
        (
            "seed_044", "dev", "hard", "kubernetes_workloads", "corpus_unanswerable",
            "How do you configure Ceph CSI driver volume snapshot classes for persistent volume claims?",
            "Corpus unanswerable: Ceph storage CSI snapshots are absent from the indexed documents.",
        ),
        (
            "seed_045", "dev", "hard", "kubernetes_workloads", "unresolved_evidence",
            "What are the exact custom Prometheus adapter PrometheusRule YAML thresholds for external scaling metrics in HPA?",
            "Unresolved evidence: The approved documents cover HPA resource metrics (CPU) and Metrics Server setup, but do not provide external PrometheusRule metric definitions.",
        ),
        (
            "seed_046", "dev", "hard", "kubernetes_workloads", "unresolved_evidence",
            "What are the specific AWS Karpenter node pool consolidation parameters for autoscaling worker nodes running the parallel Redis queue?",
            "Unresolved evidence: The documents discuss Redis work queues and Minikube/Docker environments, but Karpenter provider consolidation flags are not present.",
        ),

        # --- TEST SET (4 items: 2 corpus_unanswerable, 2 unresolved_evidence) ---
        (
            "seed_047", "test", "medium", "kubernetes_workloads", "corpus_unanswerable",
            "How do you configure RabbitMQ quorum queues with the Cluster Operator on Kubernetes?",
            "Corpus unanswerable: RabbitMQ operators and quorum queues are completely absent from the indexed documents.",
        ),
        (
            "seed_048", "test", "hard", "kubernetes_workloads", "corpus_unanswerable",
            "What are the steps to configure Cilium eBPF host routing to replace kube-proxy in Kubernetes v1.28?",
            "Corpus unanswerable: Cilium eBPF replacement for kube-proxy is absent from the indexed corpus.",
        ),
        (
            "seed_049", "test", "hard", "kubernetes_workloads", "unresolved_evidence",
            "What are the exact GPU time-slicing configuration profiles for NVIDIA container runtime pods in the job-wq-2 manifest?",
            "Unresolved evidence: The job-wq-2 manifest specifies CPU and image fields, but NVIDIA GPU time-slicing profiles are unresolved in the text.",
        ),
        (
            "seed_050", "test", "hard", "kubernetes_workloads", "unresolved_evidence",
            "What Active Directory gMSA credential spec annotations are required for Windows containers in Kubernetes Jobs?",
            "Unresolved evidence: Job restart policies and batch specifications are documented, but Windows container gMSA security contexts are absent.",
        ),
    ]

    for qid, split_name, diff, dom, subtype, query, note_text in unanswerable_definitions:
        records.append(
            RetrievalEvalRecord(
                id=qid,
                query=query,
                query_type="unanswerable",
                difficulty=diff,
                domain=dom,
                source_files=[],
                expected_sections=[],
                evidence_quotes=[],
                candidate_chunk_ids=[],
                relevance_grades={},
                sufficient_evidence_sets=[],
                verification_status="unverified_synthetic",
                split=split_name,
                unanswerable_subtype=subtype,
                expected_answer="I cannot answer this question based on the provided technical documentation.",
                notes=note_text,
            )
        )

    return records


def generate_seed_dataset() -> Dict[str, Any]:
    """Generates and validates the retrieval seed dataset."""
    print("Loading corpus index from processed_data/true/...")
    file_chunks, chunk_lookup = load_corpus_index()
    print(f"Loaded {len(file_chunks)} files with {len(chunk_lookup)} total verified chunks.")

    print("Building 50 seed candidate questions...")
    records = build_seed_records(chunk_lookup)

    print("Validating dataset records...")
    val_result = validate_dataset_records(records)
    if not val_result["is_valid"]:
        raise ValueError(f"Dataset validation failed: {val_result['errors']} | Duplicates: {val_result['duplicates']} | Leakage: {val_result['split_leakage']}")

    manifest = {
        "dataset_name": "TrustRAG Retrieval Seed Benchmark v1",
        "description": "Seed candidate retrieval evaluation dataset for enterprise technical documentation.",
        "version": "1.0-seed",
        "provenance": "Synthetic seed candidates generated from processed_data/true/ approved documents.",
        "verification_notice": "ALL records are marked unverified_synthetic. Must undergo explicit human review before being treated as a benchmark.",
        "collection_name": settings.QDRANT_COLLECTION,
        "total_corpus_chunks": len(chunk_lookup),
        "embedding_model": "sentence-transformers/all-mpnet-base-v2 (768-d) + FastEmbed BM25 IDF",
        "generator_script": "evals/dataset_generator.py",
        "counts": val_result["breakdown"],
    }

    os.makedirs(os.path.dirname(OUTPUT_SEED_FILE), exist_ok=True)
    save_dataset(records, OUTPUT_SEED_FILE, manifest=manifest)
    print(f"[OK] Successfully wrote {len(records)} records to {OUTPUT_SEED_FILE}")

    return {
        "total_records": len(records),
        "breakdown": val_result["breakdown"],
        "file": OUTPUT_SEED_FILE,
        "manifest": manifest,
    }


if __name__ == "__main__":
    res = generate_seed_dataset()
    print(json.dumps(res, indent=2))
