import os
import sys
import uuid
import json
import hashlib
from typing import Optional, List, Dict, Any
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import (
    embed_texts,
    embed_sparse_texts,
    get_embedding_dim,
)
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_docx, parse_pptx
from app.ingestion.chunking.splitter import split_units

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"
PROJECT_UUID_NAMESPACE = uuid.UUID(settings.PROJECT_UUID_NAMESPACE)

# Global Qdrant Client for ingestion
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60.0,
)


def compute_point_id(filename: str, chunk_index: int, text: str) -> str:
    """
    Computes a deterministic, idempotent UUID5 using a project-specific namespace.
    """
    content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    unique_key = f"{filename}:{chunk_index}:{content_hash}"
    return str(uuid.uuid5(PROJECT_UUID_NAMESPACE, unique_key))


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    wipe: bool = False
) -> None:
    """
    Ensures the target collection exists with named dense and FastEmbed BM25 sparse vectors.
    Sets modifier=models.Modifier.IDF on the sparse vector.
    """
    if wipe and client.collection_exists(collection_name):
        logfire.warning(f"Wiping existing collection '{collection_name}' as requested.")
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        dim = get_embedding_dim()
        logfire.info(f"Creating collection '{collection_name}' with dense ({dim}-dim) and sparse (BM25 IDF)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        logfire.info(f"✅ Collection '{collection_name}' created successfully.")

    # Ensure required payload indices exist for filtering & deletion
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        logfire.info(f"Payload index check: {e}")


MANIFEST_FILE = os.path.join(PROCESSED_DATA_DIR, "ingestion_manifest.json")


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


CHUNKING_CONFIG_VERSION = "v3.0-tokenguard-384max-15min"


def process_file(
    file_path: str,
    filename: str,
    source_type: str,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
    force_reindex: bool = False
) -> List[Dict[str, Any]]:
    """
    Parse document into page/slide/section units → split units → delete existing points for file → embed → upsert.
    Resumable via local manifest tracking:
    - file hash
    - chunking/cleaning config version
    - embedding model
    - collection name
    Changing any of these triggers re-ingestion.
    """
    target_client = client or qdrant_client
    target_collection = collection_name or settings.QDRANT_COLLECTION
    embedding_model_name = settings.DENSE_EMBEDDING_MODEL

    file_hash = get_file_hash(file_path)
    manifest_key = f"{target_collection}:{filename}:{file_hash}:{CHUNKING_CONFIG_VERSION}:{embedding_model_name}"
    manifest = load_manifest()

    if not force_reindex and manifest_key in manifest:
        entry = manifest[manifest_key]
        if (
            entry.get("status") == "completed"
            and entry.get("hash") == file_hash
            and entry.get("config_version") == CHUNKING_CONFIG_VERSION
            and entry.get("embedding_model") == embedding_model_name
            and entry.get("collection") == target_collection
        ):
            logfire.info(f"⏩ Resumable Ingest: Skipping {filename} (cached for collection='{target_collection}', config='{CHUNKING_CONFIG_VERSION}', model='{embedding_model_name}').")
            return []

    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract structured units based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                units = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                units = parse_html(file_path)
            elif ext == "txt":
                units = parse_text(file_path)
            elif ext == "docx":
                units = parse_docx(file_path)
            elif ext == "pptx":
                units = parse_pptx(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return []

            if not units or not split_units(units, chunk_size=1200, chunk_overlap=200):
                logfire.warning(f"No extractable text units/chunks from {filename} — caching in manifest as 0 chunks.")
                manifest[manifest_key] = {
                    "filename": filename,
                    "hash": file_hash,
                    "config_version": CHUNKING_CONFIG_VERSION,
                    "embedding_model": embedding_model_name,
                    "collection": target_collection,
                    "status": "completed_empty",
                    "chunks_count": 0,
                    "source_type": source_type
                }
                save_manifest(manifest)
                return []
            chunks = split_units(units, chunk_size=1200, chunk_overlap=200)

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            save_processed_locally(processed_data, source_type, filename)

            # 4. Clean out existing points for this source file to avoid stale chunks
            logfire.info(f"Purging existing points in '{target_collection}' for source='{filename}'")
            target_client.delete(
                collection_name=target_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=filename)
                            )
                        ]
                    )
                )
            )

            # 5. Embed (Dense + BM25 Sparse)
            texts = [c["text"] for c in chunks]
            dense_vectors = embed_texts(texts)
            sparse_vectors = embed_sparse_texts(texts)

            # 6. Build PointStructs with deterministic UUID5 IDs
            points = []
            for c, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
                pid = compute_point_id(filename, c["chunk_index"], c["text"])
                payload = {
                    "text": c["text"],
                    "source": filename,
                    "source_type": source_type,
                    "chunk_index": c["chunk_index"],
                    "section": c.get("section", "General"),
                }
                if "page" in c:
                    payload["page"] = c["page"]
                    if "total_pages" in c:
                        payload["total_pages"] = c["total_pages"]
                if "slide" in c:
                    payload["slide"] = c["slide"]

                points.append(
                    models.PointStruct(
                        id=pid,
                        vector={
                            "dense": dense_vec,
                            "sparse": sparse_vec,
                        },
                        payload=payload,
                    )
                )

            # 7. Upsert into Qdrant in batches with per-batch logging
            BATCH_SIZE = 50
            total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
            for b_idx in range(0, len(points), BATCH_SIZE):
                batch_points = points[b_idx : b_idx + BATCH_SIZE]
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        target_client.upsert(
                            collection_name=target_collection,
                            points=batch_points,
                        )
                        break
                    except Exception as upsert_err:
                        if attempt == max_retries - 1:
                            raise upsert_err
                        logfire.warning(f"Upsert attempt {attempt+1} failed for {filename}: {upsert_err}. Retrying in 3s...")
                        time.sleep(3.0)
                b_num = (b_idx // BATCH_SIZE) + 1
                logfire.info(f"  [Batch {b_num}/{total_batches}] Upserted {len(batch_points)} points for {filename} to '{target_collection}'")

            # 8. Record in manifest
            manifest[manifest_key] = {
                "filename": filename,
                "hash": file_hash,
                "config_version": CHUNKING_CONFIG_VERSION,
                "embedding_model": embedding_model_name,
                "collection": target_collection,
                "status": "completed",
                "chunks_count": len(points),
                "source_type": source_type
            }
            save_manifest(manifest)
            logfire.info(f"Indexed {len(points)} points to '{target_collection}' from {filename} (manifest recorded).")
            return chunks

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            raise


def process_directory(
    dir_path: str,
    source_type: str,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None
) -> None:
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(
                os.path.join(dir_path, filename),
                filename,
                source_type,
                client=client,
                collection_name=collection_name
            )


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: Optional[str] = None,
    wipe: bool = False,
    collection_name: Optional[str] = None
) -> None:
    """
    Scans base_dir and ingests all documents into the specified collection.
    """
    col_name = collection_name or settings.QDRANT_COLLECTION
    with logfire.span("Universal Ingestion Started", base_directory=base_dir, collection=col_name):
        ensure_collection_exists(qdrant_client, col_name, wipe=wipe)

        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            process_directory(base_dir, source_type, client=qdrant_client, collection_name=col_name)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(
                    os.path.join(base_dir, subdir),
                    source_type,
                    client=qdrant_client,
                    collection_name=col_name
                )


if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")
