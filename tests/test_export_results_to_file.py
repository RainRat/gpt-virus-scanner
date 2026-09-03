import json
import csv
import pytest
from gptscan import export_results_to_file

SAMPLE_RESULTS = [
    {
        "path": "src/vulnerable.py",
        "line": 42,
        "own_conf": "90%",
        "admin_desc": "Dangerous exec usage",
        "end-user_desc": "Code execution risk detected",
        "gpt_conf": "85%",
        "snippet": "exec(user_input)"
    }
]

def test_export_results_to_file_sarif(tmp_path):
    output_file = tmp_path / "results.sarif"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="sarif")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data.get("$schema")
    assert len(data.get("runs", [])) > 0

def test_export_results_to_file_html(tmp_path):
    output_file = tmp_path / "results.html"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="html")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "src/vulnerable.py" in content

def test_export_results_to_file_markdown(tmp_path):
    for fmt in ["markdown", "md"]:
        output_file = tmp_path / f"results_{fmt}.md"
        export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format=fmt)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# GPT Scan Results" in content
        assert "src/vulnerable.py" in content

def test_export_results_to_file_xml(tmp_path):
    output_file = tmp_path / "results.xml"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="xml")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert '<?xml version="1.0" encoding="utf-8"?>' in content
    assert "<findings>" in content
    assert "<path>src/vulnerable.py</path>" in content

def test_export_results_to_file_yaml(tmp_path):
    for fmt in ["yaml", "yml"]:
        output_file = tmp_path / f"results_{fmt}.yaml"
        export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format=fmt)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "path: src/vulnerable.py" in content

def test_export_results_to_file_report(tmp_path):
    output_file = tmp_path / "results.txt"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="report")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "CONSOLE TRIAGE REPORT" in content
    assert "src/vulnerable.py:42" in content

def test_export_results_to_file_json(tmp_path):
    output_file = tmp_path / "results.json"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="json")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    record = json.loads(content.strip())
    assert record["path"] == "src/vulnerable.py"

def test_export_results_to_file_csv(tmp_path):
    output_file = tmp_path / "results.csv"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="csv")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "path,own_conf,admin_desc,end-user_desc,gpt_conf,snippet,line"
    assert "src/vulnerable.py" in lines[1]

def test_export_results_to_file_tsv(tmp_path):
    output_file = tmp_path / "results.tsv"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="tsv")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "path\town_conf\tadmin_desc\tend-user_desc\tgpt_conf\tsnippet\tline"
    assert "src/vulnerable.py" in lines[1]

def test_export_results_to_file_default_fallback(tmp_path):
    output_file = tmp_path / "results.unknown"
    export_results_to_file(str(output_file), SAMPLE_RESULTS, output_format="unknown_format")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "path,own_conf,admin_desc,end-user_desc,gpt_conf,snippet,line"
