import json
import os
import pytest
from unittest.mock import MagicMock
from gptscan import generate_sarif, run_cli, Config
import gptscan

def test_sarif_round_trip(monkeypatch, tmp_path):
    """Verify that all metadata fields survive a SARIF export/import round-trip."""
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "admin_desc": "Admin Notes",
            "end-user_desc": "User Notes",
            "gpt_conf": "95%",
            "snippet": "print('hello')"
        }
    ]

    # 1. Export to SARIF
    sarif_log = generate_sarif(results)
    sarif_file = tmp_path / "roundtrip.sarif"
    sarif_file.write_text(json.dumps(sarif_log))

    # 2. Mock GUI and import back
    mock_tree = MagicMock()
    # Result tree columns
    mock_tree.__getitem__.side_effect = lambda key: (
        "path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"
    ) if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **k: str(sarif_file))
    monkeypatch.setattr(gptscan, "clear_results", MagicMock())
    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_results()

    # 3. Verify
    mock_insert.assert_called_once()
    imported_values = mock_insert.call_args[0][0]

    assert imported_values[0] == "test.py"
    assert imported_values[1] == "90%"
    assert imported_values[2] == "Admin Notes"
    assert imported_values[3] == "User Notes"
    assert imported_values[4] == "95%"
    assert imported_values[5] == "print('hello')"

def test_generate_sarif_structure():
    """Verify the overall structure of the generated SARIF log."""
    results = []
    sarif = generate_sarif(results)

    assert sarif["version"] == "2.1.0"
    assert sarif["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    tool = run["tool"]["driver"]
    assert tool["name"] == "GPT Virus Scanner"
    assert len(tool["rules"]) > 0
    assert tool["rules"][0]["id"] == "GPTScan.MaliciousContent"
    assert sarif["runs"][0]["results"] == []

def test_generate_sarif_results_mapping():
    """Verify that scan results are correctly mapped to SARIF result objects."""
    results = [
        {
            "path": "/path/to/malicious.py",
            "own_conf": "95%",
            "gpt_conf": "99%",
            "snippet": "import os; os.system('rm -rf /')",
            "admin_desc": "Dangerous system call",
            "end-user_desc": "Deletes files"
        }
    ]

    sarif = generate_sarif(results)
    sarif_results = sarif["runs"][0]["results"]

    assert len(sarif_results) == 1
    res = sarif_results[0]

    assert res["ruleId"] == "GPTScan.MaliciousContent"
    assert res["level"] == "error"  # > 80%
    assert res["message"]["text"] == "Dangerous system call"

    location = res["locations"][0]["physicalLocation"]["artifactLocation"]
    assert location["uri"] == "/path/to/malicious.py"

    props = res["properties"]
    assert props["own_conf"] == "95%"
    assert props["gpt_conf"] == "99%"
    assert props["snippet"] == "import os; os.system('rm -rf /')"

def test_generate_sarif_confidence_levels():
    """Verify logic for mapping confidence percentages to SARIF levels."""
    # level mapping: >80 -> error, >50 -> warning, else -> note
    results = [
        {"path": "file1.py", "own_conf": "85%", "admin_desc": "High risk"},
        {"path": "file2.py", "own_conf": "60%", "admin_desc": "Medium risk"},
        {"path": "file3.py", "own_conf": "40%", "admin_desc": "Low risk"},
        {"path": "file4.py", "own_conf": "invalid", "admin_desc": "Unknown risk"},
    ]

    sarif = generate_sarif(results)
    sarif_results = sarif["runs"][0]["results"]

    assert len(sarif_results) == 4
    assert sarif_results[0]["level"] == "error"
    assert sarif_results[1]["level"] == "warning"
    assert sarif_results[2]["level"] == "note"
    assert sarif_results[3]["level"] == "note" # Default on error

def test_generate_sarif_path_normalization():
    """Verify that Windows paths are normalized to URI-compatible forward slashes."""
    results = [
        {"path": "C:\\Users\\User\\script.py", "own_conf": "90%", "admin_desc": "Test"}
    ]

    sarif = generate_sarif(results)
    uri = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]

    assert uri == "C:/Users/User/script.py"

def test_run_cli_output_sarif_format(monkeypatch, capsys):
    """Integration test for run_cli with output_format='sarif'."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/file.py", "95%", "Admin Info", "User Info", "90%", "print('test')"))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    run_cli(["/dummy"], False, True, False, 60, output_format='sarif')

    captured = capsys.readouterr()
    output = captured.out.strip()

    # Ensure output is valid JSON
    try:
        sarif_log = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail("Output was not valid JSON")

    assert sarif_log["version"] == "2.1.0"
    assert len(sarif_log["runs"][0]["results"]) == 1
    assert sarif_log["runs"][0]["results"][0]["message"]["text"] == "Admin Info"

def test_import_results_sarif(monkeypatch, tmp_path):
    """Test importing a SARIF file."""
    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Suspicious behavior"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "test_script.py"
                                    }
                                }
                            }
                        ],
                        "properties": {
                            "own_conf": "85%",
                            "gpt_conf": "90%",
                            "snippet": "dangerous_code()"
                        }
                    }
                ]
            }
        ]
    }
    sarif_file = tmp_path / "results.sarif"
    sarif_file.write_text(json.dumps(sarif_data))

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(sarif_file))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, "update_status", mock_status)

    # Mock clear_results to avoid side effects
    monkeypatch.setattr(gptscan, "clear_results", MagicMock())

    gptscan.import_results()

    mock_insert.assert_called_once()
    args, _ = mock_insert.call_args
    values = args[0]
    assert values[0] == "test_script.py"
    assert values[1] == "85%"
    assert values[2] == "Suspicious behavior"
    assert values[4] == "90%"
    assert values[5] == "dangerous_code()"

    mock_status.assert_called_with(f"Imported 1 results from results.sarif")

def test_import_results_sarif_content_detection(monkeypatch, tmp_path):
    """Test that a SARIF file is detected by content even if extension is .json."""
    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {"text": "Detected threat"},
                        "locations": [{"physicalLocation": {"artifactLocation": {"uri": "file.py"}}}]
                    }
                ]
            }
        ]
    }
    json_file = tmp_path / "not_sarif_ext.json"
    json_file.write_text(json.dumps(sarif_data))

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(json_file))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    monkeypatch.setattr(gptscan, "insert_tree_row", MagicMock())
    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, "update_status", mock_status)
    monkeypatch.setattr(gptscan, "clear_results", MagicMock())

    gptscan.import_results()

    mock_status.assert_called_with(f"Imported 1 results from not_sarif_ext.json")
