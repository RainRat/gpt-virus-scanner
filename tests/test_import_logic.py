import json
import csv
import os
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_import_results_cancels(monkeypatch):
    """Test that cancelling the file dialog does nothing."""
    mock_filedialog = MagicMock()
    mock_filedialog.askopenfilename.return_value = ""
    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", mock_filedialog.askopenfilename)

    # Mock tree to ensure it's not None
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    gptscan.import_results()

    mock_tree.delete.assert_not_called()

def test_import_results_json(monkeypatch, tmp_path):
    """Test importing a standard JSON list of results."""
    data = [
        {
            "path": "test.py",
            "own_conf": "85%",
            "admin_desc": "Suspicious",
            "end-user_desc": "Don't run",
            "gpt_conf": "90%",
            "snippet": "print('hello')"
        }
    ]
    json_file = tmp_path / "results.json"
    json_file.write_text(json.dumps(data))

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(json_file))

    mock_tree = MagicMock()
    # Mocking __getitem__ to return columns when tree["columns"] is called
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, "update_status", mock_status)

    gptscan.import_results()

    mock_tree.delete.assert_called_once()
    mock_insert.assert_called_once()
    args, _ = mock_insert.call_args
    assert args[0][0] == "test.py"
    assert args[0][4] == "90%"

    mock_status.assert_called_with(f"Imported 1 results from results.json")

def test_import_results_ndjson(monkeypatch, tmp_path):
    """Test importing newline-delimited JSON (NDJSON) results."""
    line1 = {"path": "test1.py", "own_conf": "10%"}
    line2 = {"path": "test2.py", "own_conf": "20%"}
    ndjson_file = tmp_path / "results.jsonl"
    with open(ndjson_file, "w") as f:
        f.write(json.dumps(line1) + "\n")
        f.write(json.dumps(line2) + "\n")

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(ndjson_file))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_results()

    assert mock_insert.call_count == 2

def test_import_results_csv(monkeypatch, tmp_path):
    """Test importing results from a CSV file."""
    csv_file = tmp_path / "results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"])
        writer.writerow(["test.py", "50%", "Maybe", "Careful", "60%", "code"])

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(csv_file))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_results()

    mock_insert.assert_called_once()
    args, _ = mock_insert.call_args
    assert args[0][0] == "test.py"

def test_import_results_invalid_json(monkeypatch, tmp_path):
    """Test error handling when importing an invalid JSON file."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("invalid json")

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(bad_file))
    monkeypatch.setattr(gptscan, "tree", MagicMock())

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    gptscan.import_results()

    mock_messagebox.showerror.assert_called()
    # First argument to showerror is title, second is message
    assert "Import Failed" in mock_messagebox.showerror.call_args[0][0]

def test_import_results_unsupported_ext(monkeypatch, tmp_path):
    """Test error handling for unsupported file extensions."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("some text")

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **kwargs: str(txt_file))
    monkeypatch.setattr(gptscan, "tree", MagicMock())

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    gptscan.import_results()

    mock_messagebox.showerror.assert_called_with("Import Error", "Unsupported file extension: .txt")
