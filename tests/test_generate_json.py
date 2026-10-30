import os
import json
import tempfile
import pytest
import gptscan


def test_generate_json_list():
    sample_results = [
        {
            "path": "src/script.py",
            "line": "10",
            "own_conf": "85%",
            "gpt_conf": "90%",
            "admin_desc": "Dangerous eval call",
            "end-user_desc": "Suspicious execution logic",
            "snippet": "eval(user_input)",
        },
        {
            "path": "lib/utils.js",
            "line": "42",
            "own_conf": "60%",
            "gpt_conf": "",
            "admin_desc": "",
            "end-user_desc": "",
            "snippet": "exec(cmd)",
        },
    ]

    output = gptscan.generate_json(sample_results)
    parsed = json.loads(output)
    assert parsed == sample_results
    assert "  \"path\": \"src/script.py\"" in output


def test_generate_json_dict():
    sample = {
        "path": "test.py",
        "line": "1",
        "own_conf": "50%",
    }
    output = gptscan.generate_json(sample)
    parsed = json.loads(output)
    assert parsed == sample
    assert output.startswith("{\n  \"path\": \"test.py\"")


def test_generate_json_empty():
    assert gptscan.generate_json([]) == "[]"


def test_export_results_to_file_json(tmp_path):
    sample_results = [
        {
            "path": "app.py",
            "line": "5",
            "own_conf": "70%",
            "gpt_conf": "75%",
            "admin_desc": "Insecure file access",
            "end-user_desc": "Potential file leak",
            "snippet": "open(path, 'w')",
        }
    ]

    out_file = tmp_path / "report.json"
    gptscan.export_results_to_file(str(out_file), sample_results, output_format="json")

    content = out_file.read_text(encoding="utf-8").strip()
    parsed = [json.loads(line) for line in content.splitlines() if line.strip()]
    assert parsed == sample_results
    assert '"path": "app.py"' in content
