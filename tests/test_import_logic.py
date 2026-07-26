import json
import csv
import os
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_import_results_cancels(monkeypatch):
    """Test that cancelling the file dialog does nothing."""
    mock_filedialog = MagicMock()
    mock_filedialog.askopenfilenames.return_value = ()
    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", mock_filedialog.askopenfilenames)

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

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(json_file),))

    mock_tree = MagicMock()
    # Mocking __getitem__ to return columns when tree["columns"] is called
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    mock_tree.get_children.return_value = []
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, "update_status", mock_status)

    gptscan.import_results()

    mock_tree.delete.assert_not_called()
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

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(ndjson_file),))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    mock_tree.get_children.return_value = []
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

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(csv_file),))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    mock_tree.get_children.return_value = []
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_results()

    mock_insert.assert_called_once()
    args, _ = mock_insert.call_args
    assert args[0][0] == "test.py"

def test_import_results_csv_alternative_headers(monkeypatch, tmp_path):
    csv_file = tmp_path / "alt_results.csv"
    headers = ["File Path", "Local Conf.", "Admin Notes", "User Notes", "AI Conf.", "Snippet"]
    data = ["alt_test.py", "70%", "Alt Admin", "Alt User", "75%", "alt_code"]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(data)

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(csv_file),))

    mock_tree = MagicMock()
    columns = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    mock_tree.__getitem__.side_effect = lambda key: columns if key == "columns" else MagicMock()
    mock_tree.get_children.return_value = []
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    monkeypatch.setattr(gptscan, "update_status", MagicMock())
    monkeypatch.setattr(gptscan, "clear_results", MagicMock())
    monkeypatch.setattr(gptscan, "update_tree_columns", MagicMock())

    gptscan.import_results()

    mock_insert.assert_called_once()
    args, _ = mock_insert.call_args
    values = args[0]

    assert values[0] == "alt_test.py"
    assert values[1] == "70%"
    assert values[2] == "Alt Admin"
    assert values[3] == "Alt User"
    assert values[4] == "75%"
    assert values[5] == "alt_code"
    assert values[6] == ""

def test_import_results_invalid_json(monkeypatch, tmp_path):
    """Test error handling when importing an invalid JSON file."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("invalid json")

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(bad_file),))
    monkeypatch.setattr(gptscan, "tree", MagicMock())

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    gptscan.import_results()

    mock_messagebox.showerror.assert_called()
    assert "Import Errors" in mock_messagebox.showerror.call_args[0][0]

def test_import_results_unsupported_ext(monkeypatch, tmp_path):
    """Test error handling for unsupported file extensions."""
    txt_file = tmp_path / "test.xyz"
    txt_file.write_text("some text")

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(txt_file),))
    monkeypatch.setattr(gptscan, "tree", MagicMock())

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    gptscan.import_results()

    mock_messagebox.showerror.assert_called_with("Import Errors", "Failed to load some files:\ntest.xyz: Unsupported file extension: .xyz")

def test_import_results_single_object_json(tmp_path):
    """Test importing a pretty-printed single JSON object result."""
    data = {
        "path": "single.py",
        "own_conf": "95%",
        "admin_desc": "One object",
        "end-user_desc": "Careful",
        "gpt_conf": "99%",
        "snippet": "dangerous()",
        "line": "42"
    }
    json_file = tmp_path / "single.json"
    json_file.write_text(json.dumps(data, indent=2))

    results = gptscan.load_report_file(str(json_file))

    assert len(results) == 1
    assert results[0]["path"] == "single.py"
    assert results[0]["gpt_conf"] == "99%"
    assert results[0]["line"] == "42"

def test_import_results_bulk_multiple_files(monkeypatch, tmp_path):
    """Test importing multiple files at once."""
    data1 = [{"path": "file1.py", "own_conf": "10%"}]
    data2 = [{"path": "file2.py", "own_conf": "20%"}]
    file1 = tmp_path / "report1.json"
    file2 = tmp_path / "report2.json"
    file1.write_text(json.dumps(data1))
    file2.write_text(json.dumps(data2))

    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilenames", lambda **kwargs: (str(file1), str(file2)))

    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet") if key == "columns" else MagicMock()
    mock_tree.get_children.return_value = ["existing_row"] # Simulates already having items in tree
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_messagebox = MagicMock()
    mock_messagebox.askyesno.return_value = True # Select "Append"
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    mock_insert = MagicMock()
    monkeypatch.setattr(gptscan, "insert_tree_row", mock_insert)

    mock_clear = MagicMock()
    monkeypatch.setattr(gptscan, "clear_results", mock_clear)

    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, "update_status", mock_status)

    gptscan.import_results()

    mock_clear.assert_not_called()  # Since append was requested
    assert mock_insert.call_count == 2
    mock_status.assert_called_with("Appended 2 results to current list from report1.json, report2.json")

def test_import_results_directory_recursive(tmp_path):
    """Test importing results from a directory recursively."""
    sub_dir = tmp_path / "reports_dir"
    sub_dir.mkdir()

    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()

    data1 = [{"path": "nested_file.py", "own_conf": "90%", "snippet": "eval(x)"}]
    data2 = [{"path": "top_file.py", "own_conf": "50%", "snippet": "exec(y)"}]

    file1 = nested_dir / "report1.json"
    file2 = sub_dir / "report2.json"

    file1.write_text(json.dumps(data1))
    file2.write_text(json.dumps(data2))

    # Run generator with directory
    generator = gptscan.import_results_generator(str(sub_dir))
    events = list(generator)

    # Filter 'result' events
    results = [e[1] for e in events if e[0] == 'result']
    assert len(results) == 2

    # Sort by path to be deterministic
    results.sort(key=lambda x: x[0])

    assert results[0][0] == "nested_file.py"
    assert results[1][0] == "top_file.py"
