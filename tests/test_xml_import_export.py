import os
import pytest
from unittest.mock import MagicMock
import tkinter.filedialog
import gptscan

def test_generate_xml_basic():
    """Verify that generate_xml correctly outputs standard structures."""
    sample_results = [
        {
            "path": "test_script.py",
            "line": "12",
            "own_conf": "75%",
            "admin_desc": "Suspicious reverse shell script.",
            "end-user_desc": "Threat detected.",
            "gpt_conf": "90%",
            "snippet": "import socket,subprocess,os\ns=socket.socket()"
        }
    ]
    xml_output = gptscan.generate_xml(sample_results)
    assert '<?xml version="1.0" encoding="utf-8"?>' in xml_output
    assert '<findings>' in xml_output
    assert '<finding>' in xml_output
    assert '<path>test_script.py</path>' in xml_output
    assert '<line>12</line>' in xml_output
    assert '<own_conf>75%</own_conf>' in xml_output
    assert '<admin_desc>Suspicious reverse shell script.</admin_desc>' in xml_output
    assert '<end-user_desc>Threat detected.</end-user_desc>' in xml_output
    assert '<gpt_conf>90%</gpt_conf>' in xml_output
    assert '<snippet>import socket,subprocess,os\ns=socket.socket()</snippet>' in xml_output

def test_generate_xml_escaping():
    """Verify that generate_xml correctly escapes special XML characters."""
    sample_results = [
        {
            "path": "dangerous_<script>&_file.py",
            "line": "42",
            "own_conf": "99%",
            "admin_desc": "If admin_conf < 50% ignore",
            "end-user_desc": "Dangerous & harmful",
            "gpt_conf": "100%",
            "snippet": "if a < b and c > d:"
        }
    ]
    xml_output = gptscan.generate_xml(sample_results)
    assert 'dangerous_&lt;script&gt;&amp;_file.py' in xml_output
    assert 'admin_conf &lt; 50%' in xml_output
    assert 'Dangerous &amp; harmful' in xml_output
    assert 'if a &lt; b and c &gt; d:' in xml_output

def test_parse_xml_content_valid():
    """Verify that parse_xml_content reconstructs dictionaries correctly."""
    xml_input = """<?xml version="1.0" encoding="utf-8"?>
<findings>
  <finding>
    <path>a.py</path>
    <line>1</line>
    <own_conf>15%</own_conf>
    <admin_desc>Clean file</admin_desc>
    <end-user_desc>Safe</end-user_desc>
    <gpt_conf>0%</gpt_conf>
    <snippet>print("hello")</snippet>
  </finding>
</findings>
"""
    results = gptscan.parse_xml_content(xml_input)
    assert len(results) == 1
    assert results[0]["path"] == "a.py"
    assert results[0]["line"] == "1"
    assert results[0]["own_conf"] == "15%"
    assert results[0]["admin_desc"] == "Clean file"
    assert results[0]["end-user_desc"] == "Safe"
    assert results[0]["gpt_conf"] == "0%"
    assert results[0]["snippet"] == 'print("hello")'

def test_parse_xml_content_invalid():
    """Verify that parse_xml_content handles invalid or non-findings XML safely."""
    # Invalid root tag
    bad_xml_root = "<different_root><finding></finding></different_root>"
    assert gptscan.parse_xml_content(bad_xml_root) == []

    # Malformed XML syntax raises ValueError
    malformed_xml = "<findings><finding>unclosed</findings>"
    with pytest.raises(ValueError, match="Failed to parse XML content"):
        gptscan.parse_xml_content(malformed_xml)

def test_parse_report_content_auto_detect_xml():
    """Verify that parse_report_content correctly auto-detects and parses XML."""
    xml_input = """<?xml version="1.0" encoding="utf-8"?>
<findings>
  <finding>
    <path>detected.py</path>
    <line>12</line>
    <own_conf>85%</own_conf>
    <admin_desc>Admin explanation</admin_desc>
    <end-user_desc>User explanation</end-user_desc>
    <gpt_conf>95%</gpt_conf>
    <snippet>dangerous_code()</snippet>
  </finding>
</findings>
"""
    # 1. Auto-detect from <?xml header prefix
    res1 = gptscan.parse_report_content(xml_input)
    assert len(res1) == 1
    assert res1[0]["path"] == "detected.py"

    # 2. Auto-detect from <findings tag prefix
    xml_no_header = "<findings><finding><path>detected2.py</path></finding></findings>"
    res2 = gptscan.parse_report_content(xml_no_header)
    assert len(res2) == 1
    assert res2[0]["path"] == "detected2.py"

    # 3. Detection based on .xml extension hint
    res3 = gptscan.parse_report_content(xml_no_header, filename_hint="results.xml")
    assert len(res3) == 1
    assert res3[0]["path"] == "detected2.py"

def test_export_results_xml_success(monkeypatch, tmp_path):
    """Verify that results are correctly saved in XML format when exporting."""
    file_path = tmp_path / "export.xml"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(file_path))

    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    sample_results = [
        {
            "path": "test_export.py",
            "line": "5",
            "own_conf": "60%",
            "gpt_conf": "70%",
            "admin_desc": "Admin check",
            "end-user_desc": "User check",
            "snippet": "some_func()"
        }
    ]
    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", lambda item_ids: sample_results)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    gptscan.export_results()

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert '<?xml version="1.0" encoding="utf-8"?>' in content
    assert "<path>test_export.py</path>" in content
    assert "<admin_desc>Admin check</admin_desc>" in content

def test_import_results_xml_success(monkeypatch, tmp_path):
    """Verify that importing an XML file correctly populates the UI tree."""
    file_path = tmp_path / "import.xml"
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<findings>
  <finding>
    <path>imported_script.py</path>
    <line>10</line>
    <own_conf>45%</own_conf>
    <gpt_conf>55%</gpt_conf>
    <admin_desc>Admin note</admin_desc>
    <end-user_desc>User note</end-user_desc>
    <snippet>print("hello")</snippet>
  </finding>
</findings>
"""
    file_path.write_text(xml_content, encoding="utf-8")

    monkeypatch.setattr(gptscan.tkinter.filedialog, 'askopenfilenames', lambda **k: [str(file_path)])

    mock_tree = MagicMock()
    # Empty at first
    mock_tree.get_children.return_value = []
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    inserted_rows = []
    def mock_insert_row(values):
        inserted_rows.append(values)

    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert_row)
    monkeypatch.setattr(gptscan, "_auto_select_best_result", lambda: None)
    monkeypatch.setattr(gptscan, "update_tree_columns", lambda: None)

    gptscan.import_results()

    assert len(inserted_rows) == 1
    # Check that standard fields are imported correctly
    # values structure: (path, own_conf, admin_desc, end-user_desc, gpt_conf, snippet, line)
    assert inserted_rows[0][0] == "imported_script.py"
    assert inserted_rows[0][1] == "45%"
    assert inserted_rows[0][2] == "Admin note"
    assert inserted_rows[0][3] == "User note"
    assert inserted_rows[0][4] == "55%"
    assert inserted_rows[0][5] == "print(\"hello\")"
    assert inserted_rows[0][6] == "10"
