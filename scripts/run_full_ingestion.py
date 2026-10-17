import os
import sys
import time
import json
import concurrent.futures
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from app.config import settings
from app.ingestion.processor import (
    ensure_collection_exists,
    process_file,
    load_manifest,
    CHUNKING_CONFIG_VERSION,
    qdrant_client,
)

COLLECTION_NAME = settings.QDRANT_COLLECTION
TRUE_DATA_DIR = "DATA/true_data"
NOISY_DATA_DIR = "DATA/noisy_data"
RESULTS_FILE = "evals/results/ingest_v2_full_summary.json"
FILE_TIMEOUT_SECONDS = 900  # 15 minutes per file (handles large textbooks)


def process_with_timeout(fpath: str, fn: str, source_type: str, timeout: int = FILE_TIMEOUT_SECONDS):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            process_file,
            file_path=fpath,
            filename=fn,
            source_type=source_type,
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            force_reindex=False
        )
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def main():
    t_start_total = time.time()
    print(f"==================================================")
    print(f"STEP 1: INGESTION PIPELINE (v3 Config: {CHUNKING_CONFIG_VERSION})")
    print(f"Target Collection: {COLLECTION_NAME}")
    print(f"Per-file timeout: {FILE_TIMEOUT_SECONDS}s (10 min)")
    print(f"==================================================")

    # 1. Ensure collection exists
    ensure_collection_exists(qdrant_client, COLLECTION_NAME, wipe=False)

    manifest_before = load_manifest()
    print(f"Loaded existing manifest with {len(manifest_before)} entries.")

    # 2. Ingest true_data with force_reindex=False
    true_files = sorted([f for f in os.listdir(TRUE_DATA_DIR) if os.path.isfile(os.path.join(TRUE_DATA_DIR, f))])
    print(f"\n[1/3] Processing true_data ({len(true_files)} files) with force_reindex=False...")

    t0_true = time.time()
    true_chunk_counts = {}
    true_timings = {}
    for fn in true_files:
        fpath = os.path.join(TRUE_DATA_DIR, fn)
        t_f = time.time()
        try:
            chunks = process_with_timeout(fpath, fn, "true", timeout=FILE_TIMEOUT_SECONDS)
            dur = time.time() - t_f
            true_chunk_counts[fn] = len(chunks)
            true_timings[fn] = round(dur, 2)
            print(f"  Processed '{fn}': {len(chunks)} chunks in {dur:.2f}s")
        except Exception as e:
            dur = time.time() - t_f
            print(f"  ❌ Error processing true file '{fn}' ({dur:.2f}s): {e}")

    true_time = time.time() - t0_true
    total_true_chunks = sum(true_chunk_counts.values())
    print(f"✅ true_data verified/processed in {true_time:.2f}s ({total_true_chunks} total chunks).")

    # 3. Ingest noisy_data
    noisy_files = sorted([f for f in os.listdir(NOISY_DATA_DIR) if os.path.isfile(os.path.join(NOISY_DATA_DIR, f))])
    print(f"\n[2/3] Processing noisy_data ({len(noisy_files)} files) with per-batch logs and per-file timing...")

    t0_noisy = time.time()
    noisy_chunk_counts = {}
    noisy_timings = {}
    failed_files = []

    for idx, fn in enumerate(noisy_files, 1):
        fpath = os.path.join(NOISY_DATA_DIR, fn)
        t_file = time.time()
        print(f"\n[{idx}/{len(noisy_files)}] Ingesting {fn} ...", flush=True)
        try:
            chunks = process_with_timeout(fpath, fn, "noisy", timeout=FILE_TIMEOUT_SECONDS)
            dur = time.time() - t_file
            noisy_chunk_counts[fn] = len(chunks)
            noisy_timings[fn] = round(dur, 2)
            print(f"  Finished {fn}: {len(chunks)} chunks in {dur:.2f}s", flush=True)
        except concurrent.futures.TimeoutError:
            dur = time.time() - t_file
            print(f"  ❌ TIMEOUT (>10 min) on {fn} after {dur:.2f}s! Skipping and logging failure.", flush=True)
            failed_files.append({"filename": fn, "error": f"Timeout (>10 min / {dur:.1f}s)", "duration_s": round(dur, 2)})
        except Exception as e:
            dur = time.time() - t_file
            print(f"  ❌ FAILED {fn} after {dur:.2f}s: {e}. Skipping and logging failure.", flush=True)
            failed_files.append({"filename": fn, "error": str(e), "duration_s": round(dur, 2)})

    noisy_time = time.time() - t0_noisy
    total_noisy_chunks = sum(noisy_chunk_counts.values())
    print(f"\n✅ noisy_data processing completed in {noisy_time:.2f}s ({total_noisy_chunks} total chunks, {len(failed_files)} failures).")

    # 4. Verification of collection counts
    print(f"\n[3/3] Verifying Qdrant Collection State: {COLLECTION_NAME} ...")
    col_info = qdrant_client.get_collection(COLLECTION_NAME)
    total_points = col_info.points_count

    true_count_resp = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="true"))]
        ),
        exact=True
    )
    noisy_count_resp = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="noisy"))]
        ),
        exact=True
    )

    print(f"Collection Points Count: {total_points}")
    print(f"Points with source_type == 'true':  {true_count_resp.count}")
    print(f"Points with source_type == 'noisy': {noisy_count_resp.count}")

    # 5. Fetch 3 real noisy payloads
    print(f"\n--- 3 REAL NOISY PAYLOAD SAMPLES ---")
    noisy_scroll = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="noisy"))]
        ),
        limit=3,
        with_payload=True
    )[0]

    sample_payloads = []
    for i, pt in enumerate(noisy_scroll, 1):
        p = pt.payload
        sample_payloads.append({
            "id": str(pt.id),
            "source": p.get("source"),
            "source_type": p.get("source_type"),
            "chunk_index": p.get("chunk_index"),
            "section": p.get("section"),
            "page": p.get("page"),
            "text_preview": p.get("text", "")[:200]
        })
        print(f"\nSample {i}:")
        print(f"  Point ID:    {pt.id}")
        print(f"  source:      {p.get('source')}")
        print(f"  source_type: {p.get('source_type')}")
        print(f"  chunk_index: {p.get('chunk_index')}")
        print(f"  section:     {p.get('section')}")
        print(f"  page:        {p.get('page')}")
        print(f"  text_prefix: {p.get('text', '')[:120]}...")

    total_elapsed_all = time.time() - t_start_total

    # Save summary artifact
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection": COLLECTION_NAME,
        "chunking_config_version": CHUNKING_CONFIG_VERSION,
        "total_elapsed_seconds": round(total_elapsed_all, 2),
        "true_data": {
            "files_count": len(true_files),
            "chunks_count": total_true_chunks,
            "qdrant_points_count": true_count_resp.count,
            "per_file_chunks": true_chunk_counts,
            "per_file_timings_s": true_timings
        },
        "noisy_data": {
            "files_count": len(noisy_files),
            "chunks_count": total_noisy_chunks,
            "qdrant_points_count": noisy_count_resp.count,
            "per_file_chunks": noisy_chunk_counts,
            "per_file_timings_s": noisy_timings,
            "failures_count": len(failed_files),
            "failed_files": failed_files
        },
        "total_points_in_qdrant": total_points,
        "sample_noisy_payloads": sample_payloads
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved full ingestion summary to '{RESULTS_FILE}'")


if __name__ == "__main__":
    main()
