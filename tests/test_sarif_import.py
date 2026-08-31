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
