import re
import logfire
from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    """
    1. Collapse dot-leader runs (. . . . . or .......)
    2. Strip junk/non-printable characters and font CID artifacts
    3. Normalize whitespace
    """
    if not text:
        return ""
    # Collapse dot-leader runs often found in tables of contents
    text = re.sub(r"(\.\s*){3,}", " ", text)
    text = re.sub(r"\.{3,}", " ", text)
    # Strip font CID artifacts (e.g., (cid:123))
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Strip unprintable control characters (keep standard \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse repeated whitespace / line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def is_low_alpha_content(text: str, min_alpha_ratio: float = 0.40) -> bool:
    """
    Returns True if the text has very low alphabetic content (e.g. pure numbers, symbol grids, TOC leader lines).
    """
    if not text or not text.strip():
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    return (alpha_count / len(text)) < min_alpha_ratio


def parse_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF locally using pypdf.
    Cleans dot leaders, junk characters, and drops pages with very low alphabetic content (e.g. TOCs).
    Yields one unit per page so page metadata is accurate.
    Falls back to pdfplumber for pages that yield no text.
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            pages: list[dict] = []
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned = clean_pdf_text(raw_text)
                if cleaned and not is_low_alpha_content(cleaned):
                    pages.append({"page": i + 1, "text": cleaned, "total_pages": total_pages})
                elif not raw_text.strip():
                    blank_pages.append(i + 1)
                else:
                    logfire.info(f"Dropped low-alpha / TOC page {i + 1} from {file_path}.")

            # Fallback: use pdfplumber for any pages pypdf returned blank
            if blank_pages:
                logfire.info(f"pypdf returned blank on pages {blank_pages} — retrying with pdfplumber.")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            raw_fallback = page.extract_text() or ""
                            cleaned_fallback = clean_pdf_text(raw_fallback)
                            if cleaned_fallback and not is_low_alpha_content(cleaned_fallback):
                                pages.append({"page": page_num, "text": cleaned_fallback, "total_pages": total_pages})
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            # Sort by page number
            pages.sort(key=lambda p: p["page"])

            if not pages:
                logfire.warning(f"No text extracted from {file_path}. File may be fully image-based or low-alpha.")
            else:
                total_chars = sum(len(p["text"]) for p in pages)
                logfire.info(f"Extracted {len(pages)} pages ({total_chars} chars) from {file_path}.")

            return pages

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise

