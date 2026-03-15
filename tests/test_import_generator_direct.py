import json
import pytest
from gptscan import import_results_from_content_generator

def test_import_generator_json_list():
    """Test import_results_from_content_generator with a standard JSON list."""
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
    content = json.dumps(data)
    events = list(import_results_from_content_generator(content, filename_hint="test.json"))

    # Expected sequence:
    # 1. progress (Importing: test.json)
    # 2. progress (Importing: test.py)
    # 3. result (data tuple)
    # 4. summary (total=1, bytes=0, time=0.0)

    assert len(events) == 4
    assert events[0][0] == 'progress'
    assert "Importing: test.json" in events[0][1][2]

    assert events[1][0] == 'progress'
    assert "Importing: test.py" in events[1][1][2]

    assert events[2][0] == 'result'
    assert events[2][1] == ("test.py", "80%", "Suspicious", "Warning", "85%", "dangerous_code()", "10")

    assert events[3][0] == 'summary'
    assert events[3][1] == (1, 0, 0.0)

def test_import_generator_sarif():
    """Test import_results_from_content_generator with SARIF content."""
    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Threat detected"},
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
                            "snippet": "os.system('rm')"
                        }
                    }
                ]
            }
        ]
    }
    content = json.dumps(sarif_data)
    events = list(import_results_from_content_generator(content, filename_hint="test.sarif"))

    assert len(events) == 4
    assert events[2][0] == 'result'
    data = events[2][1]
    assert data[0] == "malicious.py"
    assert data[1] == "95%"
    assert data[2] == "Threat detected"
    assert data[4] == "99%"
    assert data[5] == "os.system('rm')"
    assert data[6] == "42"

def test_import_generator_csv():
    """Test import_results_from_content_generator with CSV content."""
    content = "path,own_conf,admin_desc,end-user_desc,gpt_conf,snippet,line\n" \
              "test.py,50%,Admin,User,60%,code,1"
    events = list(import_results_from_content_generator(content, filename_hint="test.csv"))

    assert len(events) == 4
    assert events[2][0] == 'result'
    data = events[2][1]
    assert data[0] == "test.py"
    assert data[1] == "50%"
    assert data[6] == "1"

def test_import_generator_malformed():
    """Test import_results_from_content_generator with malformed content."""
    content = "{invalid json"
    events = list(import_results_from_content_generator(content))

    # Should yield a progress event with an error message and then stop
    assert len(events) == 1
    assert events[0][0] == 'progress'
    assert "Error parsing report" in events[0][1][2]

def test_import_generator_empty():
    """Test import_results_from_content_generator with empty content."""
    events = list(import_results_from_content_generator(""))

    # parse_report_content returns [] for empty string
    # total = 0
    # yields 'progress' (0, 0, 'Importing: Content')
    # loop doesn't run
    # yields 'summary' (0, 0, 0.0)
    assert len(events) == 2
    assert events[0][0] == 'progress'
    assert events[1][0] == 'summary'
