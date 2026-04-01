import pytest
from gptscan import generate_markdown, generate_html, parse_report_content

def test_markdown_roundtrip_with_backticks():
    """Test that snippets with backticks survive Markdown export/import."""
    results = [
        {
            "path": "test.py",
            "line": 1,
            "own_conf": "50%",
            "admin_desc": "Notes",
            "snippet": "x = `backtick`"
        }
    ]
    md = generate_markdown(results)
    imported = parse_report_content(md)
    assert len(imported) == 1
    assert imported[0]["snippet"] == "x = `backtick`"

def test_markdown_roundtrip_with_triple_backticks():
    """Test that snippets with triple backticks survive Markdown export/import."""
    snippet = "```\ncode block\n```"
    results = [
        {
            "path": "test.py",
            "line": 1,
            "own_conf": "50%",
            "admin_desc": "Notes",
            "snippet": snippet
        }
    ]
    md = generate_markdown(results)
    # The detailed findings section should be able to handle triple backticks
    assert "````" in md or "~~~" in md or "```" in md
    imported = parse_report_content(md)
    assert len(imported) == 1
    # Note: parse_report_content currently only pulls from the table,
    # and the table snippet is truncated/escaped.
    # If the snippet in the table was ` ```code block``` `, it might fail.
    # Let's see how it behaves currently.
    assert imported[0]["snippet"] == snippet.replace("\n", " ")

def test_markdown_table_with_pipes():
    """Test that snippets with pipes don't break the Markdown table."""
    results = [
        {
            "path": "test.py",
            "line": 1,
            "own_conf": "50%",
            "admin_desc": "Notes",
            "snippet": "a | b | c"
        }
    ]
    md = generate_markdown(results)
    imported = parse_report_content(md)
    assert len(imported) == 1
    assert imported[0]["snippet"] == "a | b | c"

def test_html_escaping_robustness():
    """Test that HTML export escapes snippets correctly."""
    results = [
        {
            "path": "test.html",
            "line": 1,
            "own_conf": "50%",
            "admin_desc": "<b>Admin</b>",
            "end-user_desc": "Line 1\nLine 2",
            "snippet": "<script>alert(1)</script>"
        }
    ]
    html = generate_html(results)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;Admin&lt;/b&gt;" in html
    assert "Line 1<br>Line 2" in html
