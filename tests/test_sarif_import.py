import json
import os
from unittest.mock import MagicMock
import pytest
import gptscan

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
