"""
Benchmark embedding throughput and analyze token sequence lengths against all-mpnet-base-v2 (384 token limit).
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath("."))

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from app.config import settings

def main():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"Loading tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name)
    
    max_seq_len = model.max_seq_length
    print(f"Model max_seq_length: {max_seq_len} tokens")
    
    # 1. Fetch all real chunks from enterprise_rag_v2
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    points, _ = client.scroll(
        collection_name="enterprise_rag_v2",
        limit=500,
        with_payload=True,
        with_vectors=False
    )
    
    real_texts = [p.payload.get("text", "") for p in points]
    sources = [p.payload.get("source", "") for p in points]
    sections = [p.payload.get("section", "") for p in points]
    
    print(f"Analyzing {len(real_texts)} real chunks from true_data...")
    
    token_lengths = []
    exceeded_chunks = []
    
    for idx, (text, src, sec) in enumerate(zip(real_texts, sources, sections)):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        t_len = len(tokens)
        token_lengths.append(t_len)
        if t_len > max_seq_len:
            exceeded_chunks.append({
                "source": src,
                "section": sec,
                "token_length": t_len,
                "snippet": text[:150].replace("\n", " ")
            })
            
    avg_tokens = sum(token_lengths) / len(token_lengths)
    max_tokens = max(token_lengths)
    min_tokens = min(token_lengths)
    n_exceeded = len(exceeded_chunks)
    pct_exceeded = (n_exceeded / len(real_texts)) * 100
    
    print(f"\n=== TOKEN DISTRIBUTION ACROSS REAL CHUNKS ===")
    print(f"Total Chunks Analyzed: {len(real_texts)}")
    print(f"Average Token Length:  {avg_tokens:.1f} tokens")
    print(f"Min Token Length:      {min_tokens} tokens")
    print(f"Max Token Length:      {max_tokens} tokens")
    print(f"Chunks Exceeding {max_seq_len}: {n_exceeded} ({pct_exceeded:.1f}%)")
    
    if exceeded_chunks:
        print("\nSample Exceeded Chunks (Truncation Risk):")
        for ex in exceeded_chunks[:5]:
            print(f"  [{ex['source']}] | Section: {ex['section']} | Tokens: {ex['token_length']}")
            print(f"    Snippet: {ex['snippet']}")
            
    # 2. Benchmark warm embedding throughput (batch_size=64)
    print("\nWarming up embedding model...")
    _ = model.encode(["warmup query for torch kernels"] * 10, batch_size=10)
    
    print(f"Benchmarking warm encode throughput on {len(real_texts)} real chunks (batch_size=64)...")
    t0 = time.perf_counter()
    embeddings = model.encode(real_texts, batch_size=64, show_progress_bar=False)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    chunks_per_sec = len(real_texts) / elapsed
    print(f"Encoded {len(real_texts)} chunks in {elapsed:.2f}s -> {chunks_per_sec:.2f} chunks/sec")
    
    # 3. Estimate time for noisy_data (5,718 chunks)
    noisy_chunks = 5718
    est_sec = noisy_chunks / chunks_per_sec
    est_min = est_sec / 60
    print(f"\nEstimated time for noisy_data ({noisy_chunks} chunks): {est_min:.1f} minutes ({est_sec:.0f} seconds)")
    
    results = {
        "script": "scripts/benchmark_embeddings_tokens.py",
        "model_name": model_name,
        "max_seq_length": max_seq_len,
        "chunks_analyzed": len(real_texts),
        "avg_token_length": avg_tokens,
        "min_token_length": min_tokens,
        "max_token_length": max_tokens,
        "chunks_exceeding_384": n_exceeded,
        "pct_exceeding": pct_exceeded,
        "exceeded_samples": exceeded_chunks,
        "warm_throughput_chunks_per_sec": chunks_per_sec,
        "elapsed_seconds": elapsed,
        "noisy_corpus_chunks": noisy_chunks,
        "noisy_corpus_estimated_minutes": est_min
    }
    
    with open("evals/results/embedding_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results to evals/results/embedding_benchmark.json")

if __name__ == "__main__":
    main()
