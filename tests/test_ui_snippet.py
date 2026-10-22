import pytest
from ui.app import truncate_snippet

def test_truncate_snippet_short_unchanged():
    text = "Short text under 400 characters."
    assert truncate_snippet(text, max_chars=400) == text

def test_truncate_snippet_line_boundary():
    line1 = "A" * 100
    line2 = "B" * 100
    line3 = "C" * 300
    text = f"{line1}\n{line2}\n{line3}"
    truncated = truncate_snippet(text, max_chars=250)
    assert truncated.startswith(f"{line1}\n{line2}")
    assert "..." in truncated
    assert len(truncated) <= 300

def test_truncate_snippet_closes_unclosed_code_fence():
    code_inside = "\n".join([f"line_{i} = {i}" for i in range(50)])
    text = f"Intro header:\n```python\n{code_inside}\n```\nFooter text"
    truncated = truncate_snippet(text, max_chars=150)
    # Must have closed the fence
    assert truncated.count("```") == 2
    assert truncated.endswith("```\n...")
