import logfire

def parse_text(file_path: str) -> list[dict]:
    """
    Parses plain text / markdown files into structured units split by headers.
    Yields {"section": section_title, "text": section_text}.
    """
    with logfire.span("📄 Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.splitlines()
            units = []
            current_section = "Overview"
            current_lines = []
            in_code_block = False

            for line in lines:
                stripped = line.strip()

                # Toggle fenced code block
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    current_lines.append(line)
                    continue

                # Markdown headers must not be inside code blocks and must start at column 0 (not indented)
                is_markdown_header = (
                    not in_code_block
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and (line.startswith("# ") or line.startswith("## ") or line.startswith("### "))
                )

                if is_markdown_header:
                    if current_lines:
                        txt = "\n".join(current_lines).strip()
                        if txt:
                            units.append({"section": current_section, "text": txt})
                        current_lines = []
                    current_section = stripped.lstrip("#").strip()
                elif (stripped.startswith("===") or stripped.startswith("---")) and not in_code_block:
                    # Previous line was likely a header
                    if len(current_lines) >= 1 and current_lines[-1].strip():
                        prev_header = current_lines.pop().strip()
                        if current_lines:
                            txt = "\n".join(current_lines).strip()
                            if txt:
                                units.append({"section": current_section, "text": txt})
                            current_lines = []
                        current_section = prev_header
                else:
                    current_lines.append(line)
            
            if current_lines:
                txt = "\n".join(current_lines).strip()
                if txt:
                    units.append({"section": current_section, "text": txt})
            
            if not units and content.strip():
                units.append({"section": "Overview", "text": content.strip()})
                
            logfire.info(f"Extracted {len(units)} sections from {file_path}")
            return units
        except Exception as e:
            logfire.error(f"❌ Text Parse Failed: {e}")
            raise e
