import json
import pytest
import gptscan


def test_generate_json_formatting():
    results = [
        {
            "path": "test.py",
            "own_conf": "85%",
            "admin_desc": "Admin note",
            "end-user_desc": "User note",
            "gpt_conf": "90%",
            "snippet": "import os",
            "line": 10
        }
    ]
    json_str = gptscan.generate_json(results)
    assert json_str.startswith("[") and json_str.endswith("]")
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["path"] == "test.py"
    assert parsed[0]["own_conf"] == "85%"


def test_generate_json_empty():
    results = []
    json_str = gptscan.generate_json(results)
    assert json_str.strip() == "[]"
    parsed = json.loads(json_str)
    assert parsed == []
