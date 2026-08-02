import os
import sys
import pytest
from unittest.mock import MagicMock
import tkinter.filedialog
import gptscan

def test_generate_yaml_basic():
    """Verify that generate_yaml correctly outputs standard structures."""
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
    yaml_output = gptscan.generate_yaml(sample_results)
    assert 'test_script.py' in yaml_output
    assert 'own_conf: 75%' in yaml_output
    assert 'admin_desc: Suspicious reverse shell script.' in yaml_output

def test_parse_yaml_content_valid_list():
    """Verify that parse_yaml_content reconstructs dictionaries correctly when input is a list."""
    yaml_input = """
- path: a.py
  line: "1"
  own_conf: 15%
  admin_desc: Clean file
  end-user_desc: Safe
  gpt_conf: 0%
  snippet: print("hello")
"""
    results = gptscan.parse_yaml_content(yaml_input)
    assert len(results) == 1
    assert results[0]["path"] == "a.py"
    assert results[0]["line"] == "1"
    assert results[0]["own_conf"] == "15%"
    assert results[0]["admin_desc"] == "Clean file"
    assert results[0]["end-user_desc"] == "Safe"
    assert results[0]["gpt_conf"] == "0%"
    assert results[0]["snippet"] == 'print("hello")'

def test_parse_yaml_content_valid_dict():
    """Verify that parse_yaml_content reconstructs dictionaries correctly when input is a single dictionary."""
    yaml_input = """
path: b.py
line: "2"
own_conf: 80%
admin_desc: Suspicious
end-user_desc: Threat
gpt_conf: 90%
snippet: eval(x)
"""
    results = gptscan.parse_yaml_content(yaml_input)
    assert len(results) == 1
    assert results[0]["path"] == "b.py"
    assert results[0]["line"] == "2"
    assert results[0]["snippet"] == "eval(x)"

def test_parse_yaml_content_invalid():
    """Verify that parse_yaml_content handles invalid syntax safely."""
    # Malformed YAML syntax raises ValueError
    malformed_yaml = "{"
    with pytest.raises(ValueError, match="Failed to parse YAML content"):
        gptscan.parse_yaml_content(malformed_yaml)

def test_parse_report_content_auto_detect_yaml():
    """Verify that parse_report_content correctly auto-detects and parses YAML."""
    yaml_input = """
- path: detected_yaml.py
  line: "12"
  own_conf: 85%
  admin_desc: YAML admin explanation
  end-user_desc: YAML user explanation
  gpt_conf: 95%
  snippet: dangerous_yaml_code()
"""
    # 1. Auto-detect from 'path:' substring check
    res1 = gptscan.parse_report_content(yaml_input)
    assert len(res1) == 1
    assert res1[0]["path"] == "detected_yaml.py"

    # 2. Detection based on .yaml / .yml extension hint
    res2 = gptscan.parse_report_content(yaml_input, filename_hint="results.yaml")
    assert len(res2) == 1
    assert res2[0]["path"] == "detected_yaml.py"

    res3 = gptscan.parse_report_content(yaml_input, filename_hint="results.yml")
    assert len(res3) == 1
    assert res3[0]["path"] == "detected_yaml.py"

def test_export_results_yaml_success(monkeypatch, tmp_path):
    """Verify that results are correctly saved in YAML format when exporting."""
    file_path = tmp_path / "export.yaml"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(file_path))

    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    sample_results = [
        {
            "path": "test_export.py",
            "line": "5",
            "own_conf": "60%",
            "gpt_conf": "70%",
            "admin_desc": "Admin YAML check",
            "end-user_desc": "User YAML check",
            "snippet": "some_yaml_func()"
        }
    ]
    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", lambda item_ids: sample_results)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    gptscan.export_results()

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "test_export.py" in content
    assert "Admin YAML check" in content

def test_import_results_yaml_success(monkeypatch, tmp_path):
    """Verify that importing a YAML file correctly populates the UI tree."""
    file_path = tmp_path / "import.yaml"
    yaml_content = """
- path: imported_yaml_script.py
  line: "10"
  own_conf: 45%
  gpt_conf: 55%
  admin_desc: Admin YAML note
  end-user_desc: User YAML note
  snippet: print("yaml")
"""
    file_path.write_text(yaml_content, encoding="utf-8")

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
    assert inserted_rows[0][0] == "imported_yaml_script.py"
    assert inserted_rows[0][1] == "45%"
    assert inserted_rows[0][2] == "Admin YAML note"
    assert inserted_rows[0][3] == "User YAML note"
    assert inserted_rows[0][4] == "55%"
    assert inserted_rows[0][5] == "print(\"yaml\")"
    assert inserted_rows[0][6] == "10"

def test_yaml_export_import_missing_pyyaml(monkeypatch):
    """Verify that generate_yaml and parse_yaml_content raise ImportError when yaml is missing."""
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError("mocked import error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="PyYAML is required"):
        gptscan.generate_yaml([])

    with pytest.raises(ImportError, match="PyYAML is required"):
        gptscan.parse_yaml_content("")
