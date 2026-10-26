# TrustRAG Golden Dataset Audit Report (v2)

**Target Dataset**: `evals/data/retrieval_golden_seed_v1.json`  
**Total Records**: 50  
**Corpus Total Approved Chunks**: 197 (from `processed_data/true/`)  
**Unique Candidate Chunk UUIDs Referenced**: 35  
**Qdrant Verification**: Found 35 of 35 in `enterprise_rag_v2`  

---

## 1. Executive Summary

| Category | Count | Status | Action Required |
| :--- | :--- | :--- | :--- |
| **Definite Errors** | 0 | PASS | Immediate remediation |
| **Suspected Issues** | 63 | WARNING | Review and document |
| **Potential Semantic Duplicates** | 0 | REVIEW | Check question diversity |
| **Pending Human Review** | 50 | PENDING | Human adjudication required |

---

## 2. Definite Errors

*No definite schema errors, invalid UUIDs, broken references, or conjunction integrity violations found.*

---

## 3. Suspected Issues

- **Record `seed_004`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '134fb45c-5ffa-5fd2-b5b8-02c405f097d9'.
- **Record `seed_005`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '1c470929-b8ba-52dc-afd7-ff41adcd2672'.
- **Record `seed_006`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '81a4f786-ef7a-52e5-9af0-339d7dd76240'.
- **Record `seed_007`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'dd8b3b09-eaf6-59dd-b6d3-1844d69b8901'.
- **Record `seed_008`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '28cb15bb-064d-5f01-a0fc-0b0c718ed336'.
- **Record `seed_009`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'c0f1220e-829c-50f0-bf0d-2e48be167355'.
- **Record `seed_010`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '6abf9610-2c0e-589b-b750-9721ae48209d'.
- **Record `seed_012`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'acb1333f-abc9-5354-aefd-d1c7e5363ac7'.
- **Record `seed_013`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'ae33514d-f6dd-5196-88fb-5c2bdfd486ca'.
- **Record `seed_014`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '6d56e8c5-bbe7-5b5c-bfc9-d70f78860f8a'.
- **Record `seed_015`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'd7ffbe9f-96d5-5b67-97a6-f427a438db9f'.
- **Record `seed_017`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'b78e305c-b8e8-5dbb-83c5-1c82aa074a81'.
- **Record `seed_018`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '162d277b-bab0-5ec4-8394-8c2d4e995485'.
- **Record `seed_021`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '035b47c7-23b2-50c0-ba48-3fc7c1618743'.
- **Record `seed_022`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '978f78e6-b72f-5e0e-b3f5-02537347e6f9'.
- **Record `seed_023`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '0caf42f8-acd1-5ba1-a3cb-cfb255ec39d0'.
- **Record `seed_024`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '641b2b5f-33d3-5c99-9bdf-73aaa6b5c7a0'.
- **Record `seed_025`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '951973ef-00b5-56e7-b8f5-fd4e902592a5'.
- **Record `seed_026`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'a0679ae4-768b-5707-ab7e-5d2c4c497374'.
- **Record `seed_027`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '3b74d3dc-cc5e-5278-86f6-e8f62f11bbfb'.
- **Record `seed_028`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '970d2701-aab8-5f97-902b-438917febcf0'.
- **Record `seed_029`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'f8f23e79-1072-5d9e-a18e-48335d357ab2'.
- **Record `seed_030`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '65050403-7244-5c49-9ec2-4f8e52cc101e'.
- **Record `seed_031`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'd731bfd3-be36-57c0-827f-d1c85339dac3'.
- **Record `seed_031`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '38bfbdef-fa3f-542a-b5c0-6f739d9004cc'.
- **Record `seed_031`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '83358e9c-431c-51b4-97e8-baacb2d11344'.
- **Record `seed_031`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '83358e9c-431c-51b4-97e8-baacb2d11344'.
- **Record `seed_032`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '134fb45c-5ffa-5fd2-b5b8-02c405f097d9'.
- **Record `seed_032`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '81a4f786-ef7a-52e5-9af0-339d7dd76240'.
- **Record `seed_032`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '81a4f786-ef7a-52e5-9af0-339d7dd76240'.
- **Record `seed_033`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'dd8b3b09-eaf6-59dd-b6d3-1844d69b8901'.
- **Record `seed_033`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'dd8b3b09-eaf6-59dd-b6d3-1844d69b8901'.
- **Record `seed_033`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '28cb15bb-064d-5f01-a0fc-0b0c718ed336'.
- **Record `seed_033`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '28cb15bb-064d-5f01-a0fc-0b0c718ed336'.
- **Record `seed_034`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'c0f1220e-829c-50f0-bf0d-2e48be167355'.
- **Record `seed_034`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '6abf9610-2c0e-589b-b750-9721ae48209d'.
- **Record `seed_034`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '6abf9610-2c0e-589b-b750-9721ae48209d'.
- **Record `seed_034`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '4c8a8363-c0ff-51a5-8d81-3a2451d86e18'.
- **Record `seed_034`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '4c8a8363-c0ff-51a5-8d81-3a2451d86e18'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'a0679ae4-768b-5707-ab7e-5d2c4c497374'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'a0679ae4-768b-5707-ab7e-5d2c4c497374'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '6d56e8c5-bbe7-5b5c-bfc9-d70f78860f8a'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '6d56e8c5-bbe7-5b5c-bfc9-d70f78860f8a'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'acb1333f-abc9-5354-aefd-d1c7e5363ac7'.
- **Record `seed_035`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'acb1333f-abc9-5354-aefd-d1c7e5363ac7'.
- **Record `seed_036`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'b78e305c-b8e8-5dbb-83c5-1c82aa074a81'.
- **Record `seed_036`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'b78e305c-b8e8-5dbb-83c5-1c82aa074a81'.
- **Record `seed_036`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'ec8addca-27cd-551c-ae24-160f9dbd48d9'.
- **Record `seed_036`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'ec8addca-27cd-551c-ae24-160f9dbd48d9'.
- **Record `seed_037`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '035b47c7-23b2-50c0-ba48-3fc7c1618743'.
- **Record `seed_037`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '035b47c7-23b2-50c0-ba48-3fc7c1618743'.
- **Record `seed_037`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '978f78e6-b72f-5e0e-b3f5-02537347e6f9'.
- **Record `seed_037`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '978f78e6-b72f-5e0e-b3f5-02537347e6f9'.
- **Record `seed_038`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'e073ea7e-0849-56f7-ad12-f65a1b4f98ab'.
- **Record `seed_038`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '641b2b5f-33d3-5c99-9bdf-73aaa6b5c7a0'.
- **Record `seed_038`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk '641b2b5f-33d3-5c99-9bdf-73aaa6b5c7a0'.
- **Record `seed_039`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'd7ffbe9f-96d5-5b67-97a6-f427a438db9f'.
- **Record `seed_039`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk '3dd9d259-0b91-5980-9630-5951638d7349'.
- **Record `seed_039`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'aaa73a7c-a388-55be-9693-2f14f69dc961'.
- **Record `seed_039`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'aaa73a7c-a388-55be-9693-2f14f69dc961'.
- **Record `seed_040`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'acb1333f-abc9-5354-aefd-d1c7e5363ac7'.
- **Record `seed_040`** [quote_not_exact_substring]: Evidence quote [1] is not an exact substring of chunk 'ae33514d-f6dd-5196-88fb-5c2bdfd486ca'.
- **Record `seed_040`** [quote_not_exact_substring]: Evidence quote [2] is not an exact substring of chunk 'ae33514d-f6dd-5196-88fb-5c2bdfd486ca'.

---

## 4. Potential Semantic Duplicates

*No high-similarity duplicate pairs detected.*

---

## 5. Human Review Requirement

All **50 records** currently remain marked as `unverified_synthetic`.
No record may be promoted to `verified_human` or exported into the benchmark freeze without explicit review recording:
1. Reviewer identity
2. UTC timestamp
3. Documented rationale
4. Evidence verification (support, partial_support, contradiction, irrelevant)
5. Expected answer verification
