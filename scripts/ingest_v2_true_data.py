import os
import sys
import time
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import settings
from app.ingestion.processor import (
    ensure_collection_exists,
    process_file,
    compute_point_id,
    qdrant_client
)

COLLECTION_NAME = "enterprise_rag_v2"
TRUE_DATA_DIR = "DATA/true_data"

print(f"=== INGESTING TRUE DATA INTO '{COLLECTION_NAME}' ===")
print("Constraint: Do NOT touch noisy_data.")
print(f"Target Collection: {COLLECTION_NAME}")

# 1. Create collection with Dense (768-d) and Sparse (BM25 IDF) if not already existing
ensure_collection_exists(qdrant_client, COLLECTION_NAME, wipe=False)

# 2. Ingest true_data files
files = sorted([f for f in os.listdir(TRUE_DATA_DIR) if os.path.isfile(os.path.join(TRUE_DATA_DIR, f))])
print(f"Found {len(files)} files in {TRUE_DATA_DIR}: {files}")

t0 = time.time()
total_chunks = 0
for filename in files:
    fpath = os.path.join(TRUE_DATA_DIR, filename)
    print(f"\nProcessing {filename}...")
    chunks = process_file(
        file_path=fpath,
        filename=filename,
        source_type="true",
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        force_reindex=True  # Ensure fresh indexing into v2
    )
    print(f"  -> Generated and indexed {len(chunks)} chunks for {filename}")
    total_chunks += len(chunks)

elapsed = time.time() - t0
print(f"\n✅ Ingestion of true_data finished in {elapsed:.2f}s ({total_chunks} total chunks)")

# 3. Verify Point Counts in Qdrant
info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
print(f"\n=== QDRANT VERIFICATION FOR '{COLLECTION_NAME}' ===")
print(f"Points count in collection: {info.points_count}")
print(f"Vectors config: {info.config.params.vectors}")
print(f"Sparse vectors config: {info.config.params.sparse_vectors}")

# 4. Count by source
count_resp = qdrant_client.count(
    collection_name=COLLECTION_NAME,
    count_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value="true")
            )
        ]
    ),
    exact=True
)
print(f"Exact count with source_type == 'true': {count_resp.count}")

# 5. Fetch sample payloads across different file formats
print("\n=== SAMPLE PAYLOADS BY FORMAT ===")
for ext in ["html", "docx", "pptx", "txt"]:
    target_files = [f for f in files if f.endswith(f".{ext}")]
    if not target_files:
        continue
    sample_file = target_files[0]
    pts = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=sample_file))]
        ),
        limit=1,
        with_payload=True
    )[0]
    if pts:
        p = pts[0]
        payload = p.payload
        print(f"\n--- Format: .{ext.upper()} | Source: {sample_file} ---")
        print(f"Point ID:    {p.id}")
        print(f"source:      {payload.get('source')}")
        print(f"source_type: {payload.get('source_type')}")
        print(f"chunk_index: {payload.get('chunk_index')}")
        print(f"section:     {payload.get('section')}")
        if "page" in payload:
            print(f"page:        {payload.get('page')}")
        if "slide" in payload:
            print(f"slide:       {payload.get('slide')}")
        print(f"text sample: {payload.get('text', '')[:120]}...")
