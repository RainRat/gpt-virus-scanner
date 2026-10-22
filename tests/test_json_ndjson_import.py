import json
import pytest
import gptscan


def test_parse_report_content_json_array():
    json_data = [
        {
            "path": "test1.py",
            "line": 10,
            "own_conf": "85%",
            "gpt_conf": "90%",
            "admin_desc": "Admin note 1",
            "end-user_desc": "User note 1",
            "snippet": "import os"
        },
        {
            "path": "test2.js",
            "line": 1,
            "own_conf": "10%",
            "gpt_conf": "0%",
            "admin_desc": "Safe file",
            "end-user_desc": "Safe",
            "snippet": "console.log('hi')"
        }
    ]
    content = json.dumps(json_data)
    results = gptscan.parse_report_content(content)
    assert len(results) == 2
    assert results[0]["path"] == "test1.py"
    assert results[0]["own_conf"] == "85%"
    assert results[1]["path"] == "test2.js"


def test_parse_report_content_json_single_object():
    json_obj = {
        "path": "test_single.py",
        "line": 5,
        "own_conf": "95%",
        "admin_desc": "Critical malware",
        "snippet": "eval(data)"
    }
    content = json.dumps(json_obj)
    results = gptscan.parse_report_content(content)
    assert len(results) == 1
    assert results[0]["path"] == "test_single.py"
    assert results[0]["own_conf"] == "95%"


def test_parse_report_content_ndjson_autodetect():
    lines = [
        json.dumps({"path": "file1.py", "own_conf": "80%", "snippet": "exec(x)"}),
        json.dumps({"path": "file2.sh", "own_conf": "70%", "snippet": "rm -rf /"}),
    ]
    content = "\n".join(lines)
    results = gptscan.parse_report_content(content)
    assert len(results) == 2
    assert results[0]["path"] == "file1.py"
    assert results[1]["path"] == "file2.sh"


def test_parse_report_content_json_hints():
    content = json.dumps([{"path": "hint.py", "own_conf": "60%"}])
    results_json = gptscan.parse_report_content(content, filename_hint="report.json")
    assert len(results_json) == 1
    assert results_json[0]["path"] == "hint.py"

    ndjson_content = json.dumps({"path": "hint.py", "own_conf": "60%"}) + "\n" + json.dumps({"path": "hint2.py", "own_conf": "50%"})
    results_ndjson = gptscan.parse_report_content(ndjson_content, filename_hint="report.ndjson")
    assert len(results_ndjson) == 2
    assert results_ndjson[1]["path"] == "hint2.py"
