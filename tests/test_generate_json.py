import json
import os
import tempfile
import pytest
import gptscan


def test_generate_json_empty():
    res = gptscan.generate_json([])
    assert res == "[]"
    assert json.loads(res) == []


def test_generate_json_dict():
    data = {
        "path": "test.py",
        "own_conf": "80%",
        "admin_desc": "Suspicious pattern",
        "end-user_desc": "Low risk",
        "gpt_conf": "20%",
        "snippet": "import os",
        "line": "10"
    }
    res = gptscan.generate_json(data)
    parsed = json.loads(res)
    assert parsed == data


def test_generate_json_list():
    results = [
        {
            "path": "file1.py",
            "own_conf": "90%",
            "admin_desc": "High threat",
            "end-user_desc": "Danger",
            "gpt_conf": "85%",
            "snippet": "os.system('rm -rf /')",
            "line": "5"
        },
        {
            "path": "file2.py",
            "own_conf": "10%",
            "admin_desc": "Safe",
            "end-user_desc": "Safe",
            "gpt_conf": "5%",
            "snippet": "print('hello')",
            "line": "1"
        }
    ]
    res = gptscan.generate_json(results)
    parsed = json.loads(res)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["path"] == "file1.py"
    assert parsed[1]["path"] == "file2.py"


def test_export_results_to_file_json():
    results = [
        {
            "path": "sample.py",
            "own_conf": "50%",
            "admin_desc": "Check",
            "end-user_desc": "Check",
            "gpt_conf": "50%",
            "snippet": "eval(x)",
            "line": "12"
        }
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
        temp_path = tf.name

    try:
        gptscan.export_results_to_file(temp_path, results, output_format='json')
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = json.loads(content)
        assert len(parsed) == 1
        assert parsed[0]["path"] == "sample.py"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
