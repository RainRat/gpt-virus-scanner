import pytest
from gptscan import parse_report_content

def test_import_markdown_table():
    md_content = """
# GPT Scan Results

## Summary Table

| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| script.py | 10 | 85% | **Admin:** Malicious code found <br> **User:** Potential threat | `print("malicious")` |
| test.js | 5 | 40% | **Admin:** Safe script | `console.log("safe")` |
"""
    results = parse_report_content(md_content, filename_hint="report.md")
    assert len(results) == 2

    assert results[0]["path"] == "script.py"
    assert results[0]["line"] == "10"
    assert results[0]["own_conf"] == "85%"
    assert results[0]["admin_desc"] == "Malicious code found"
    assert results[0]["end-user_desc"] == "Potential threat"
    assert results[0]["snippet"] == 'print("malicious")'

    assert results[1]["path"] == "test.js"
    assert results[1]["line"] == "5"
    assert results[1]["own_conf"] == "40%"
    assert results[1]["admin_desc"] == "Safe script"
    assert results[1]["end-user_desc"] == ""
    assert results[1]["snippet"] == 'console.log("safe")'

def test_import_markdown_escaped_pipe():
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| file\\|pipe.py | 1 | 50% | **Admin:** contains \\| pipe | `a \\| b` |
"""
    results = parse_report_content(md_content)
    assert len(results) == 1
    assert results[0]["path"] == "file|pipe.py"
    assert results[0]["admin_desc"] == "contains | pipe"
    assert results[0]["snippet"] == "a | b"

def test_import_markdown_no_extension_hint():
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| script.py | 10 | 85% | **Admin:** Malicious | `code` |
"""
    # Should detect via '|' even without hint
    results = parse_report_content(md_content)
    assert len(results) == 1
    assert results[0]["path"] == "script.py"
