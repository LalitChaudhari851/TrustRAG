"""
Analyze token lengths across DATA/noisy_data before vs after token guard.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))

from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_docx, parse_pptx
from app.ingestion.chunking.splitter import count_tokens, is_low_alpha_chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    data_dir = "DATA/noisy_data"
    files = sorted(os.listdir(data_dir))
    
    file_loaders = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".pptx": parse_pptx,
        ".html": parse_html,
        ".htm": parse_html,
        ".txt": parse_text,
    }
    
    raw_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    
    total_raw_chunks = 0
    raw_over_384 = 0
    raw_under_15 = 0
    
    over_samples = []
    under_samples = []
    
    print(f"Scanning 58 noisy files in {data_dir} for token length violations...")
    
    for f in files:
        ext = os.path.splitext(f)[-1].lower()
        loader = file_loaders.get(ext)
        if not loader:
            continue
        path = os.path.join(data_dir, f)
        units = loader(path)
        
        for u in units:
            text = u.get("text", "").strip()
            if not text:
                continue
            sub_texts = raw_splitter.split_text(text)
            for st in sub_texts:
                st = st.strip()
                if not st or is_low_alpha_chunk(st):
                    continue
                total_raw_chunks += 1
                n_tok = count_tokens(st)
                if n_tok > 384:
                    raw_over_384 += 1
                    if len(over_samples) < 5:
                        over_samples.append({"file": f, "tokens": n_tok, "snippet": st[:100].replace("\n", " ")})
                elif n_tok < 15:
                    raw_under_15 += 1
                    if len(under_samples) < 5:
                        under_samples.append({"file": f, "tokens": n_tok, "snippet": st[:100].replace("\n", " ")})
                        
    pct_over = (raw_over_384 / total_raw_chunks) * 100 if total_raw_chunks else 0
    pct_under = (raw_under_15 / total_raw_chunks) * 100 if total_raw_chunks else 0
    
    print("\n=== RAW NOISY DATA TOKEN DISTRIBUTION (BEFORE TOKEN GUARD) ===")
    print(f"Total Raw Chunks:        {total_raw_chunks}")
    print(f"Chunks Exceeding 384:    {raw_over_384} ({pct_over:.2f}%)")
    print(f"Chunks Under 15 Tokens:  {raw_under_15} ({pct_under:.2f}%)")
    
    print("\nSample Over-384 Chunks:")
    for s in over_samples:
        print(f"  [{s['file']}] Tokens: {s['tokens']} | {s['snippet']}...")
        
    print("\nSample Under-15 Chunks:")
    for s in under_samples:
        print(f"  [{s['file']}] Tokens: {s['tokens']} | {s['snippet']}...")
        
    results = {
        "script": "scripts/analyze_noisy_tokens_before.py",
        "total_raw_chunks": total_raw_chunks,
        "chunks_exceeding_384": raw_over_384,
        "pct_exceeding_384": pct_over,
        "chunks_under_15": raw_under_15,
        "pct_under_15": pct_under,
        "sample_over_384": over_samples,
        "sample_under_15": under_samples
    }
    
    with open("evals/results/noisy_tokens_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to evals/results/noisy_tokens_analysis.json")

if __name__ == "__main__":
    main()
