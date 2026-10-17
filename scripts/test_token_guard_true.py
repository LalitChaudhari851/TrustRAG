"""
Test token guard on DATA/true_data and verify the 6 previously over-long chunks in cronjobs.docx.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))

from app.ingestion.loaders.office import parse_docx, parse_pptx
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import split_units, count_tokens

def main():
    true_dir = "DATA/true_data"
    files = sorted(os.listdir(true_dir))
    
    file_loaders = {
        ".docx": parse_docx,
        ".pptx": parse_pptx,
        ".html": parse_html,
        ".htm": parse_html,
        ".txt": parse_text,
    }
    
    all_results = {}
    total_before = 212  # baseline without token guard
    total_after = 0
    over_384_after = 0
    under_15_after = 0
    
    cronjobs_chunks = []
    
    for f in files:
        ext = os.path.splitext(f)[-1].lower()
        loader = file_loaders.get(ext)
        if not loader:
            continue
        path = os.path.join(true_dir, f)
        units = loader(path)
        chunks = split_units(units, chunk_size=1200, chunk_overlap=200, max_tokens=384, min_tokens=15)
        
        token_lens = [count_tokens(c["text"]) for c in chunks]
        total_after += len(chunks)
        over = sum(1 for tl in token_lens if tl > 384)
        under = sum(1 for tl in token_lens if tl < 15)
        over_384_after += over
        under_15_after += under
        
        all_results[f] = {
            "units": len(units),
            "chunks_after": len(chunks),
            "token_lengths_min": min(token_lens) if token_lens else 0,
            "token_lengths_max": max(token_lens) if token_lens else 0,
            "token_lengths_avg": round(sum(token_lens)/len(token_lens), 1) if token_lens else 0,
            "over_384": over,
            "under_15": under
        }
        
        if f == "cronjobs.docx":
            cronjobs_chunks = chunks
            
    print("=== TOKEN GUARD EVALUATION ON TRUE_DATA ===")
    print(f"Total Chunks Before: {total_before}")
    print(f"Total Chunks After:  {total_after}")
    print(f"Chunks Exceeding 384 Tokens: {over_384_after}")
    print(f"Chunks Under 15 Tokens:      {under_15_after}")
    print("\nPer File Breakdown:")
    for f, stats in all_results.items():
        print(f"  {f:<25}: {stats['chunks_after']:>3} chunks (min={stats['token_lengths_min']}, max={stats['token_lengths_max']}, avg={stats['token_lengths_avg']})")
        
    # Inspect previously over-long sections in cronjobs.docx
    target_sections = [
        "1. Job State Backup",
        "3. Data Retention Policies",
        "3. Data Recovery Procedures",
        "2. Job Testing Framework"
    ]
    
    print("\n=== INSPECTING RESOLVED SECTIONS IN CRONJOBS.DOCX ===")
    resolved_samples = []
    for c in cronjobs_chunks:
        sec = c.get("section", "")
        if any(ts in sec for ts in target_sections):
            tok_count = count_tokens(c["text"])
            resolved_samples.append({
                "section": sec,
                "chunk_index": c["chunk_index"],
                "token_count": tok_count,
                "snippet": c["text"][:120].replace("\n", " ")
            })
            print(f"  Section: {sec} | Tokens: {tok_count} (<= 384: {tok_count <= 384})")
            print(f"    Snippet: {c['text'][:100].replace(chr(10), ' ')}...")
            
    output_payload = {
        "script": "scripts/test_token_guard_true.py",
        "total_chunks_before": total_before,
        "total_chunks_after": total_after,
        "chunks_exceeding_384": over_384_after,
        "chunks_under_15": under_15_after,
        "per_file": all_results,
        "resolved_target_sections": resolved_samples
    }
    
    with open("evals/results/token_guard_true_results.json", "w", encoding="utf-8") as out_f:
        json.dump(output_payload, out_f, indent=2)
    print("\nSaved to evals/results/token_guard_true_results.json")

if __name__ == "__main__":
    main()
