import os
import logfire
import docx
import pptx

def parse_docx(file_path: str) -> list[dict]:
    """
    Parses DOCX documents into structured units split by section headings.
    Yields {"section": heading_text, "text": section_text}.
    """
    with logfire.span("📄 DOCX Parsing", filename=file_path):
        doc = docx.Document(file_path)
        units = []
        current_heading = "Overview"
        current_paras = []

        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            if p.style and p.style.name and p.style.name.startswith("Heading"):
                if current_paras:
                    units.append({"section": current_heading, "text": "\n".join(current_paras)})
                    current_paras = []
                current_heading = txt
            else:
                current_paras.append(txt)

        if current_paras:
            units.append({"section": current_heading, "text": "\n".join(current_paras)})

        # Also capture tables if present
        for t_idx, table in enumerate(doc.tables):
            table_rows = []
            for row in table.rows:
                row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                units.append({
                    "section": f"Table {t_idx + 1}",
                    "text": "\n".join(table_rows)
                })

        logfire.info(f"Extracted {len(units)} sections from {file_path}")
        return units


def parse_pptx(file_path: str) -> list[dict]:
    """
    Parses PPTX presentations into structured units per slide.
    Yields {"slide": slide_num, "slide_title": title, "text": slide_text}.
    """
    with logfire.span("📄 PPTX Parsing", filename=file_path):
        prs = pptx.Presentation(file_path)
        units = []

        for i, slide in enumerate(prs.slides):
            title = (
                slide.shapes.title.text.strip()
                if slide.shapes.title and slide.shapes.title.text
                else f"Slide {i + 1}"
            )
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        t = paragraph.text.strip()
                        if t:
                            texts.append(t)
            slide_text = "\n".join(texts)
            if slide_text.strip():
                units.append({
                    "slide": i + 1,
                    "slide_title": title,
                    "text": slide_text
                })

        logfire.info(f"Extracted {len(units)} slides from {file_path}")
        return units


def parse_office(file_path: str) -> list[dict]:
    """
    Dispatches office document parsing by extension (.docx or .pptx).
    """
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "docx":
        return parse_docx(file_path)
    elif ext == "pptx":
        return parse_pptx(file_path)
    else:
        raise ValueError(f"Unsupported office document format: {file_path}")
