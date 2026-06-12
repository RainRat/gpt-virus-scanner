import pytest
from gptscan import parse_report_content

def test_markdown_import_varied_whitespace():
    """Test parsing Markdown table with varied whitespace in Analysis column."""
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| test.py | 1 | 90% | **Admin:**   Notes with spaces   <br>   **User:**   More spaces | `snippet` |
"""
    results = parse_report_content(md_content)
    assert len(results) == 1
    assert results[0]["admin_desc"] == "Notes with spaces"
    assert results[0]["end-user_desc"] == "More spaces"

def test_markdown_import_missing_labels():
    """Test parsing Markdown table where labels might be missing in Analysis column."""
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| no_user.py | 1 | 50% | **Admin:** Only admin | `code` |
| no_admin.py | 2 | 40% | **User:** Only user | `code` |
| empty.py | 3 | 10% |  | `code` |
"""
    results = parse_report_content(md_content)
    assert len(results) == 3

    assert results[0]["path"] == "no_user.py"
    assert results[0]["admin_desc"] == "Only admin"
    assert results[0]["end-user_desc"] == ""

    assert results[1]["path"] == "no_admin.py"
    assert results[1]["admin_desc"] == ""
    assert results[1]["end-user_desc"] == "Only user"

    assert results[2]["path"] == "empty.py"
    assert results[2]["admin_desc"] == ""
    assert results[2]["end-user_desc"] == ""

def test_markdown_import_multiline_notes():
    """Test parsing Markdown table with multiline notes using <br>."""
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| multi.py | 1 | 95% | **Admin:** Line 1<br>Line 2<br>Line 3 <br> **User:** User Line 1<br>User Line 2 | `code` |
"""
    results = parse_report_content(md_content)
    assert len(results) == 1
    assert results[0]["admin_desc"] == "Line 1\nLine 2\nLine 3"
    assert results[0]["end-user_desc"] == "User Line 1\nUser Line 2"

def test_markdown_import_escaped_entities():
    """Test parsing Markdown table with escaped pipes and HTML entities."""
    md_content = """
| Path | Line | Threat Level | Analysis | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| path\\|with\\|pipes.py | 1 | 50% | **Admin:** &lt;script&gt; tag with \\| pipe | `a &amp;&amp; b \\|\\| c` |
"""
    results = parse_report_content(md_content)
    assert len(results) == 1
    assert results[0]["path"] == "path|with|pipes.py"
    assert results[0]["admin_desc"] == "<script> tag with | pipe"
    assert results[0]["snippet"] == "a && b || c"

def test_markdown_import_messy_alignment():
    """Test parsing Markdown table with messy column alignment."""
    md_content = """
|Path|Line|Threat Level|Analysis|Snippet|
|---|---|---|---|---|
|messy.py|10|80%|**Admin:**notes<br>**User:**user|`snippet`|
|  spaced.py  |  20  |  10%  |  **Admin:**  notes  |  `code`  |
"""
    results = parse_report_content(md_content)
    assert len(results) == 2
    assert results[0]["path"] == "messy.py"
    assert results[0]["admin_desc"] == "notes"
    assert results[0]["end-user_desc"] == "user"
    assert results[1]["path"] == "spaced.py"
    assert results[1]["admin_desc"] == "notes"
    assert results[1]["end-user_desc"] == ""
