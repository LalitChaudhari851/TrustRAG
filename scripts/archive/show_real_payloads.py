import json
import sys
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.office import parse_docx, parse_pptx
from app.ingestion.loaders.html import parse_html
from app.ingestion.chunking.splitter import split_units

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. HTML: job_management.html
html_units = parse_html("DATA/true_data/job_management.html")
html_chunks = split_units(html_units, chunk_size=1200, chunk_overlap=200)
# Find the chunk with the CLI command
cli_chunk = None
for c in html_chunks:
    if "databricks jobs get" in c["text"]:
        cli_chunk = c
        break
if not cli_chunk:
    cli_chunk = html_chunks[3]

html_payload = {
    "source": "job_management.html",
    "source_type": "true",
    "chunk_index": cli_chunk["chunk_index"],
    "section": cli_chunk["section"],
    "text": cli_chunk["text"]
}

# 2. DOCX: cronjobs.docx
docx_units = parse_docx("DATA/true_data/cronjobs.docx")
docx_chunks = split_units(docx_units, chunk_size=1200, chunk_overlap=200)
sample_docx = docx_chunks[2]
docx_payload = {
    "source": "cronjobs.docx",
    "source_type": "true",
    "chunk_index": sample_docx["chunk_index"],
    "section": sample_docx["section"],
    "text": sample_docx["text"]
}

# 3. PPTX: architecture.pptx
pptx_units = parse_pptx("DATA/true_data/architecture.pptx")
pptx_chunks = split_units(pptx_units, chunk_size=1200, chunk_overlap=200)
sample_pptx = pptx_chunks[1]
pptx_payload = {
    "source": "architecture.pptx",
    "source_type": "true",
    "chunk_index": sample_pptx["chunk_index"],
    "section": sample_pptx["section"],
    "slide": sample_pptx["slide"],
    "text": sample_pptx["text"]
}

# 4. PDF: A Malloc Tutorial (Feb 16, 2009).pdf
pdf_units = parse_pdf("DATA/noisy_data/A Malloc Tutorial (Feb 16, 2009).pdf")
pdf_chunks = split_units(pdf_units, chunk_size=1200, chunk_overlap=200)
sample_pdf = pdf_chunks[0]
pdf_payload = {
    "source": "A Malloc Tutorial (Feb 16, 2009).pdf",
    "source_type": "noisy",
    "chunk_index": sample_pdf["chunk_index"],
    "section": sample_pdf["section"],
    "page": sample_pdf["page"],
    "total_pages": sample_pdf["total_pages"],
    "text": sample_pdf["text"]
}

print("=== REAL PAYLOAD: HTML (WITH PRE/CODE PRESERVED) ===")
print(json.dumps(html_payload, indent=2))

print("\n=== REAL PAYLOAD: DOCX ===")
print(json.dumps(docx_payload, indent=2))

print("\n=== REAL PAYLOAD: PPTX ===")
print(json.dumps(pptx_payload, indent=2))

print("\n=== REAL PAYLOAD: PDF ===")
print(json.dumps(pdf_payload, indent=2))
