# TrustRAG Golden Dataset Split Audit Report (v2)

**Target Dataset**: `evals/data/retrieval_golden_seed_v1.json`  
**Split Counts**: Dev = 30, Test = 20  
**Shared Candidate Chunks**: 6 chunks shared between Dev and Test  
**Shared Source Files**: 6 of 6 files shared across splits  
**Verbatim Query Leakage**: 0  

---

## 1. Split Risk Assessment

- **Verbatim Query Leakage**: `ZERO` (0 verbatim queries shared)
- **Chunk-Level Leakage**: `HIGH` (6 exact chunk IDs referenced in both dev and test)
- **Document-Level Leakage**: `CRITICAL` (100% of source files appear in both dev and test)

> **Policy Recommendation**: Current partition uses random query-level splitting across all 6 documents. For an enterprise benchmark resistant to memorization, a document-disjoint split (e.g. 4 documents dev / 2 documents test) or cluster-disjoint split is strongly recommended.

---

## 2. Shared Candidate Chunks Detail

| Chunk UUID | Source Document | Dev Queries | Test Queries | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| `3dd9d259...` | `monitor_job.docx` | `seed_016` | `seed_039` | Now let's start Minikube to create a local Kubernetes cluster:
minikube start --driver=docker
This command will take a f... |
| `a0679ae4...` | `architecture.pptx` | `seed_035` | `seed_026` | etcd
Etcd acts as the cluster datastore; providing a strong, consistent and highly  available key-value store used for p... |
| `acb1333f...` | `architecture.pptx` | `seed_012`, `seed_035` | `seed_040` | Master Components
Kube-apiserver
Etcd
Kube-controller-manager
Cloud-controller-manager
Kube-scheduler... |
| `ae33514d...` | `architecture.pptx` | `seed_013` | `seed_040` | kube-apiserver
The apiserver provides a forward facing REST interface into the kubernetes  control plane and datastore. ... |
| `d7ffbe9f...` | `monitor_job.docx` | `seed_015` | `seed_039` | Minikube is typically pre-installed on many lab environments. Let's verify the installation by checking the version:
min... |
| `e073ea7e...` | `pods_autoscale.html` | `seed_011` | `seed_038` | Create a file named hpa.yaml with the following content:

```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
m... |

---

## 3. Source Document Distribution

| Document Name | Dev Queries | Test Queries | Shared Across Splits? |
| :--- | :--- | :--- | :--- |
| `architecture.pptx` | 4 | 3 | **YES (SHARED)** |
| `cronjobs.docx` | 7 | 4 | **YES (SHARED)** |
| `job_management.html` | 3 | 1 | **YES (SHARED)** |
| `monitor_job.docx` | 2 | 3 | **YES (SHARED)** |
| `parallel_work_queue.txt` | 4 | 2 | **YES (SHARED)** |
| `pods_autoscale.html` | 4 | 3 | **YES (SHARED)** |

---

## 4. Cross-Split Semantic Similarities

*No high cross-split query similarities found.*
