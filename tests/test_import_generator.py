import pytest
import json
import os
from gptscan import import_results_generator

@pytest.fixture
def temp_json_report(tmp_path):
    report_file = tmp_path / "report.json"
    data = [
        {
            "path": "test.py",
            "own_conf": "80%",
            "admin_desc": "Suspicious",
            "end-user_desc": "Warning",
            "gpt_conf": "85%",
            "snippet": "dangerous_code()",
            "line": "10"
        }
    ]
    report_file.write_text(json.dumps(data))
    return str(report_file)

def test_import_generator_sarif(temp_sarif_report):
    """Test import_results_generator with a SARIF report."""
    events = list(import_results_generator(temp_sarif_report))

    # Verify result event mapping
    # 0: progress (loading)
    # 1: progress (item)
    # 2: result
    # 3: summary
    result_event = events[2]
    assert result_event[0] == 'result'
    data = result_event[1]
    assert data[0] == "malicious.py"
    assert data[1] == "95%"
    assert data[2] == "Detected threat"
    assert data[4] == "99%"
    assert data[5] == "os.system('rm -rf /')"
    assert data[6] == "42"

def test_import_generator_empty_file(tmp_path):
    """Test import_results_generator with an empty file."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")
    events = list(import_results_generator(str(empty_file)))

    # Should yield a progress event with an error message
    assert events[0][0] == 'progress'
    assert "Error" in events[0][1][2]

def test_import_generator_invalid_format(tmp_path):
    """Test import_results_generator with an invalid JSON file."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid: json}")
    events = list(import_results_generator(str(bad_file)))

    assert events[0][0] == 'progress'
    assert "Error" in events[0][1][2]

def test_import_generator_non_existent_file():
    """Test import_results_generator with a non-existent file."""
    events = list(import_results_generator("non_existent.json"))

    assert events[0][0] == 'progress'
    assert "Error" in events[0][1][2]

def test_import_generator_json(temp_json_report):
    """Test import_results_generator with a standard JSON report."""
    events = list(import_results_generator(temp_json_report))

    # Verify progress events
    assert events[0][0] == 'progress'
    assert events[1][0] == 'progress'

    # Verify result event
    result_event = events[2]
    assert result_event[0] == 'result'
    data = result_event[1]
    assert data[0] == "test.py"
    assert data[1] == "80%"
    assert data[2] == "Suspicious"
    assert data[3] == "Warning"
    assert data[4] == "85%"
    assert data[5] == "dangerous_code()"
    assert data[6] == "10"

    # Verify summary event
    summary_event = events[3]
    assert summary_event[0] == 'summary'
    assert summary_event[1][0] == 1  # Total results

@pytest.fixture
def temp_sarif_report(tmp_path):
    report_file = tmp_path / "report.sarif"
    data = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Detected threat"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "malicious.py"},
                                    "region": {"startLine": 42}
                                }
                            }
                        ],
                        "properties": {
                            "own_conf": "95%",
                            "gpt_conf": "99%",
                            "snippet": "os.system('rm -rf /')"
                        }
                    }
                ]
            }
        ]
    }
    report_file.write_text(json.dumps(data))
    return str(report_file)
