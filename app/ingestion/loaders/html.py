from bs4 import BeautifulSoup 
import logfire

def parse_html(file_path: str) -> list[dict]:
    """
    Parses HTML content into structured units split by section headings.
    Preserves <pre>, <code>, <table>, <ol>, <ul>, and <p> blocks in exact document order.
    Yields list of {"section": heading_text, "text": section_text}.
    """
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove scripts, styles, metadata, navigation, footers
            for junk in soup(["script", "style", "meta", "noscript", "nav", "footer"]):
                junk.decompose()
            
            body = soup.body or soup
            default_heading = soup.title.string.strip() if soup.title and soup.title.string else "Overview"
            current_heading = default_heading
            
            units = []
            current_blocks = []

            def flush_section():
                nonlocal current_blocks, current_heading
                if current_blocks:
                    combined = "\n\n".join(current_blocks).strip()
                    if combined:
                        units.append({"section": current_heading, "text": combined})
                    current_blocks = []

            # Traverse top-level content elements in document order
            for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol", "table"]):
                # Skip tags that are nested inside another collected block
                if tag.find_parent(["pre", "ul", "ol", "table"]):
                    continue
                
                tag_name = tag.name.lower()
                if tag_name in ["h1", "h2", "h3", "h4"]:
                    flush_section()
                    heading_text = tag.get_text().strip()
                    current_heading = heading_text if heading_text else default_heading
                elif tag_name == "pre":
                    pre_text = tag.get_text().strip()
                    if pre_text:
                        current_blocks.append(f"```\n{pre_text}\n```")
                elif tag_name == "table":
                    rows = []
                    for tr in tag.find_all("tr"):
                        cells = [c.get_text().strip() for c in tr.find_all(["th", "td"])]
                        if cells:
                            rows.append(" | ".join(cells))
                    if rows:
                        current_blocks.append("\n".join(rows))
                elif tag_name in ["ul", "ol"]:
                    items = [f"- {li.get_text().strip()}" for li in tag.find_all("li", recursive=False) if li.get_text().strip()]
                    if items:
                        current_blocks.append("\n".join(items))
                elif tag_name == "p":
                    p_text = tag.get_text().strip()
                    if p_text:
                        current_blocks.append(p_text)
            
            flush_section()

            if not units:
                text = soup.get_text(separator="\n")
                lines = (line.strip() for line in text.splitlines())
                clean_text = "\n".join(l for l in lines if l)
                if clean_text:
                    units.append({"section": default_heading, "text": clean_text})
            
            logfire.info(f"Extracted {len(units)} sections from {file_path}")
            return units
        except Exception as e:
            logfire.error(f"❌ HTML Parse Failed: {e}")
            raise e
