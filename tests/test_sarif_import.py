"""Tests for SARIF report import functionality in gptscan."""

import json
import pytest
import gptscan


def test_parse_report_content_sarif_auto_detect():
    """Test auto-detection and parsing of SARIF JSON strings without .sarif extension hint."""
    sarif_data = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "GPTScan"
                    }
                },
                "results": [
                    {
                        "ruleId": "GPTScan.MaliciousContent",
                        "level": "error",
                        "message": {
                            "text": "Suspicious eval execution"
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "src/danger.py"
                                    },
                                    "region": {
                                        "startLine": 42
                                    }
                                }
                            }
                        ],
                        "properties": {
                            "own_conf": "85%",
                            "gpt_conf": "90%",
                            "admin_desc": "Evaluates user string directly",
                            "end-user_desc": "Contains dangerous eval statement",
                            "snippet": "eval(user_input)"
                        }
                    }
                ]
            }
        ]
    }
    content = json.dumps(sarif_data)

    results = gptscan.parse_report_content(content)
    assert len(results) == 1
    res = results[0]
    assert res["path"].replace("\\", "/") == "src/danger.py"
    assert str(res["line"]) == "42"
    assert res["own_conf"] == "85%"
    assert res["gpt_conf"] == "90%"
    assert res["admin_desc"] == "Evaluates user string directly"
    assert res["end-user_desc"] == "Contains dangerous eval statement"
    assert res["snippet"] == "eval(user_input)"


def test_import_results_generator_sarif(tmp_path):
    """Test importing SARIF file via generator."""
    sarif_file = tmp_path / "scan_output.sarif"
    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Backdoor pattern"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "lib/backdoor.js"},
                                    "region": {"startLine": 12}
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    sarif_file.write_text(json.dumps(sarif_data), encoding="utf-8")

    events = list(gptscan.import_results_generator(str(sarif_file)))
    results = [data for evt_type, data in events if evt_type == 'result']

    assert len(results) == 1
    # Check result tuple format: (path, own_conf, admin_desc, user_desc, gpt_conf, snippet, line)
    path, own_conf, admin_desc, user_desc, gpt_conf, snippet, line = results[0]
    assert "backdoor.js" in path
    assert admin_desc == "Backdoor pattern"
    assert str(line) == "12"


def test_parse_sarif_content_empty_input():
    assert gptscan.parse_sarif_content({}) == []
    assert gptscan.parse_sarif_content({"runs": []}) == []
    assert gptscan.parse_sarif_content({"runs": [{"results": []}]}) == []


def test_parse_sarif_content_missing_locations_and_properties():
    sarif_minimal = {
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Generic finding message"}
                    }
                ]
            }
        ]
    }
    res_minimal = gptscan.parse_sarif_content(sarif_minimal)
    assert len(res_minimal) == 1
    assert res_minimal[0] == {
        "path": "",
        "own_conf": "",
        "admin_desc": "Generic finding message",
        "end-user_desc": "",
        "gpt_conf": "",
        "snippet": "",
        "line": "-"
    }


def test_parse_sarif_content_properties_override_message():
    sarif_override = {
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Fallback message"},
                        "locations": [],
                        "properties": {
                            "admin_desc": "Explicit admin note",
                            "own_conf": "70%",
                            "end-user_desc": "User summary",
                            "gpt_conf": "80%",
                            "snippet": "var a = 1;"
                        }
                    }
                ]
            }
        ]
    }
    res_override = gptscan.parse_sarif_content(sarif_override)
    assert len(res_override) == 1
    assert res_override[0]["path"] == ""
    assert res_override[0]["line"] == "-"
    assert res_override[0]["admin_desc"] == "Explicit admin note"
    assert res_override[0]["own_conf"] == "70%"
    assert res_override[0]["end-user_desc"] == "User summary"
    assert res_override[0]["gpt_conf"] == "80%"
    assert res_override[0]["snippet"] == "var a = 1;"


def test_parse_sarif_content_missing_start_line():
    sarif_no_start_line = {
        "runs": [
            {
                "results": [
                    {
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "src/app.py"},
                                    "region": {"endLine": 10}
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    res_no_start_line = gptscan.parse_sarif_content(sarif_no_start_line)
    assert len(res_no_start_line) == 1
    assert res_no_start_line[0]["path"] in ("src/app.py", "src\\app.py")
    assert res_no_start_line[0]["line"] == "-"
