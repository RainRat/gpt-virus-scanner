import json
import pytest
import gptscan


def test_generate_json_empty():
    results = []
    output = gptscan.generate_json(results)
    assert output == "[]"
    assert json.loads(output) == []


def test_generate_json_with_items():
    results = [
        {
            "path": "/test/path/script.py",
            "own_conf": "85%",
            "admin_desc": "Suspicious execution of shell command",
            "end-user_desc": "Potential malware script",
            "gpt_conf": "90%",
            "snippet": "os.system('rm -rf /')",
            "line": 10
        }
    ]
    output = gptscan.generate_json(results)
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["path"] == "/test/path/script.py"
    assert parsed[0]["own_conf"] == "85%"
    assert "os.system" in parsed[0]["snippet"]
