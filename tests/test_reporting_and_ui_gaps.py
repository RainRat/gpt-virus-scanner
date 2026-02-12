
import json
import csv
import os
import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_import_results_edge_cases(monkeypatch):
    """Test import_results with missing tree and empty data."""
    # 1. Tree is None
    monkeypatch.setattr(gptscan, "tree", None)
    # Should return early without error
    assert gptscan.import_results() is None

    # 2. Empty file
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Mock file dialog to return a dummy path
    monkeypatch.setattr(gptscan.tkinter.filedialog, "askopenfilename", lambda **k: "empty.json")

    # Mock open to return empty content
    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: "")))):
        mock_msgbox = MagicMock()
        monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)
        gptscan.import_results()
        # Should show error "File is empty."
        mock_msgbox.showerror.assert_called()
        args, _ = mock_msgbox.showerror.call_args
        assert "File is empty." in args[1]

    # 3. No data found in file
    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: "[]")))):
        mock_msgbox = MagicMock()
        monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)
        gptscan.import_results()
        mock_msgbox.showwarning.assert_called_with("Import Warning", "No data found in the selected file.")

def test_export_results_edge_cases(monkeypatch, tmp_path):
    """Test export_results with missing tree and SARIF format."""
    # 1. Tree is None
    monkeypatch.setattr(gptscan, "tree", None)
    assert gptscan.export_results() is None

    # 2. SARIF Export success
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    mock_tree.get_children.return_value = ["item1"]

    original_values = ("test.py", "90%", "Admin", "User", "95%", "print('hello\nworld')")
    mock_tree.item.return_value = {"values": list(original_values) + [json.dumps(original_values)]}

    monkeypatch.setattr(gptscan, "tree", mock_tree)

    sarif_path = tmp_path / "results.sarif"
    monkeypatch.setattr(gptscan.tkinter.filedialog, "asksaveasfilename", lambda **k: str(sarif_path))
    monkeypatch.setattr(gptscan.messagebox, "showinfo", MagicMock())

    gptscan.export_results()

    assert sarif_path.exists()
    with open(sarif_path) as f:
        sarif_data = json.load(f)
        assert sarif_data["version"] == "2.1.0"
        # Check that newline was preserved in snippet
        assert sarif_data["runs"][0]["results"][0]["properties"]["snippet"] == "print('hello\nworld')"

def test_ui_helpers_no_selection(monkeypatch):
    """Test UI helpers when no selection is made in the Treeview."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # These should return early without doing anything
    assert gptscan.open_file() is None
    assert gptscan.copy_path() is None
    assert gptscan.copy_snippet() is None
    assert gptscan.show_in_folder() is None

    mock_tree.item.assert_not_called()

def test_adjust_newlines_preservation():
    """Test that adjust_newlines preserves existing newlines while wrapping long lines."""
    # Simple measure function: each char is 1 unit
    def measure(text): return len(text)

    # 1. Preserve original newlines
    val = "line1\nline2"
    result = gptscan.adjust_newlines(val, width=100, pad=0, measure=measure)
    assert result == "line1\nline2"

    # 2. Wrap long lines but keep existing ones
    val = "this is a very long line that should be wrapped\nshort line"
    # width 20, words: this(4) is(2) a(1) very(4) long(4) line(4) -> "this is a very long" (18)
    # next line: "line that should be" ...
    result = gptscan.adjust_newlines(val, width=20, pad=0, measure=measure)
    assert "this is a very long\nline that should be" in result
    assert "short line" in result

    # 3. Empty lines preserved
    val = "line1\n\nline3"
    result = gptscan.adjust_newlines(val, width=100, pad=0, measure=measure)
    assert result == "line1\n\nline3"

def test_clipboard_data_integrity(monkeypatch):
    """Test that copy_path and copy_snippet use original un-wrapped data."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1"]

    # Snippet with spaces and newlines to ensure hidden JSON isn't mangled by wrapping
    original_snippet = "def hello():\n    print('hello world')"
    original_path = "some/long/path/to/file.py"
    original_values = (original_path, "10%", "admin", "user", "0%", original_snippet)

    # Simulated Treeview values (with wrapping and hidden column)
    wrapped_snippet = "def hello():\n    print('hello\nworld')"
    wrapped_path = "some/long\npath/to/file.py"

    # Hidden JSON column
    vals_list = [wrapped_path, "10%", "admin", "user", "0%", wrapped_snippet, json.dumps(original_values)]

    def mock_item(iid, option=None):
        if option == "values":
            return vals_list
        return {"values": vals_list}

    mock_tree.item.side_effect = mock_item
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Test copy_path
    gptscan.copy_path()
    mock_tree.clipboard_append.assert_called_with(original_path)

    # Test copy_snippet
    mock_tree.clipboard_append.reset_mock()
    gptscan.copy_snippet()
    mock_tree.clipboard_append.assert_called_with(original_snippet)
