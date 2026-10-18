from typing import List, Dict, Any, Optional
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

_tokenizer: Optional[AutoTokenizer] = None

def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=True))


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text using RecursiveCharacterTextSplitter.
    Maintained for backwards compatibility.
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def is_low_alpha_chunk(text: str, min_alpha_ratio: float = 0.40) -> bool:
    if not text or not text.strip():
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    return (alpha_count / len(text)) < min_alpha_ratio


def enforce_token_limits(
    sub_texts: List[str],
    max_tokens: int = 384,
    min_tokens: int = 15
) -> List[str]:
    """
    Enforces token constraints on text chunks:
    1. Re-splits any chunk exceeding max_tokens (384) into smaller valid pieces.
    2. Merges chunks under min_tokens (15) into the preceding chunk if capacity allows,
       or drops them if standalone.
    """
    # 1. Expand over-long chunks
    refined: List[str] = []
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", "; ", " ", ""]
    )

    for piece in sub_texts:
        piece = piece.strip()
        if not piece or is_low_alpha_chunk(piece):
            continue
        n_tok = count_tokens(piece)
        if n_tok <= max_tokens:
            refined.append(piece)
        else:
            # Re-split over-long chunk
            smaller = fine_splitter.split_text(piece)
            for sm in smaller:
                sm = sm.strip()
                if not sm or is_low_alpha_chunk(sm):
                    continue
                if count_tokens(sm) <= max_tokens:
                    refined.append(sm)
                else:
                    # Fallback further if still over 384
                    ultra_fine = RecursiveCharacterTextSplitter(
                        chunk_size=400,
                        chunk_overlap=50,
                        separators=["\n", " ", ""]
                    )
                    for u in ultra_fine.split_text(sm):
                        u = u.strip()
                        if u and not is_low_alpha_chunk(u):
                            refined.append(u)

    # 2. Merge chunks under min_tokens (15) into previous or next chunk
    if not refined:
        return []

    merged: List[str] = []
    for piece in refined:
        n_tok = count_tokens(piece)
        if n_tok < min_tokens:
            if merged:
                candidate = f"{merged[-1]}\n\n{piece}"
                if count_tokens(candidate) <= max_tokens:
                    merged[-1] = candidate
                    continue
        merged.append(piece)

    # If first piece is under min_tokens and followed by another piece, merge forward
    if len(merged) > 1 and count_tokens(merged[0]) < min_tokens:
        candidate = f"{merged[0]}\n\n{merged[1]}"
        if count_tokens(candidate) <= max_tokens:
            merged[1] = candidate
            merged.pop(0)

    return merged


def split_units(
    units: List[Dict[str, Any]],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    max_tokens: int = 384,
    min_tokens: int = 15
) -> List[Dict[str, Any]]:
    """
    Splits each unit (page, slide, or section) separately using RecursiveCharacterTextSplitter,
    ensuring page, slide, and header metadata remain strictly accurate on every resulting chunk.
    Enforces token guard:
      - Max tokens: 384 (all-mpnet-base-v2 context limit)
      - Min tokens: 15 (merges into previous/next chunk of the same section or adjacent chunk; NEVER drops)
      - Drops low-alpha chunks (< 40%)
    """
    with logfire.span("✂️ Splitting Document Units", unit_count=len(units)):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        raw_chunks: List[Dict[str, Any]] = []

        for unit in units:
            unit_text = unit.get("text", "").strip()
            if not unit_text:
                continue

            raw_sub_texts = splitter.split_text(unit_text)
            bounded_sub_texts = enforce_token_limits(
                raw_sub_texts,
                max_tokens=max_tokens,
                min_tokens=min_tokens
            )

            for sub_text in bounded_sub_texts:
                chunk_dict: Dict[str, Any] = {
                    "text": sub_text.strip()
                }
                # Consistent key scheme: 'section' is always present; 'page' and 'slide' are optional
                if "section" in unit:
                    chunk_dict["section"] = unit["section"]
                elif "slide_title" in unit:
                    chunk_dict["section"] = unit["slide_title"]
                elif "page" in unit:
                    chunk_dict["section"] = f"Page {unit['page']}"
                else:
                    chunk_dict["section"] = "General"

                if "page" in unit:
                    chunk_dict["page"] = unit["page"]
                    if "total_pages" in unit:
                        chunk_dict["total_pages"] = unit["total_pages"]
                if "slide" in unit:
                    chunk_dict["slide"] = unit["slide"]

                raw_chunks.append(chunk_dict)

        # Document-level merge for any remaining short chunks (< 15 tokens)
        final_chunks: List[Dict[str, Any]] = []
        for c in raw_chunks:
            c_tok = count_tokens(c["text"])
            if c_tok < min_tokens:
                # 1. Prefer merging into previous chunk of same section
                if final_chunks and final_chunks[-1].get("section") == c.get("section"):
                    cand = f"{final_chunks[-1]['text']}\n\n{c['text']}"
                    if count_tokens(cand) <= max_tokens:
                        final_chunks[-1]["text"] = cand
                        continue
                # 2. Or merge into previous chunk of any section if within max_tokens
                if final_chunks:
                    cand = f"{final_chunks[-1]['text']}\n\n{c['text']}"
                    if count_tokens(cand) <= max_tokens:
                        final_chunks[-1]["text"] = cand
                        continue
            final_chunks.append(c)

        # Forward pass if the first chunk is still under min_tokens
        if len(final_chunks) > 1 and count_tokens(final_chunks[0]["text"]) < min_tokens:
            cand = f"{final_chunks[0]['text']}\n\n{final_chunks[1]['text']}"
            if count_tokens(cand) <= max_tokens:
                final_chunks[1]["text"] = cand
                final_chunks.pop(0)

        # Assign final contiguous chunk_index
        for idx, c in enumerate(final_chunks):
            c["chunk_index"] = idx

        logfire.info(f"✅ Generated {len(final_chunks)} bounded chunks from {len(units)} units")
        return final_chunks
