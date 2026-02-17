import pytest
from unittest.mock import MagicMock, patch
import json
import os
import gptscan
from pathlib import Path

@pytest.fixture
def mock_gui_globals(monkeypatch):
    mock_tree = MagicMock()
    mock_filter_var = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)
    monkeypatch.setattr(gptscan, "filter_var", mock_filter_var)
    monkeypatch.setattr(gptscan, "_all_results_cache", [])

    # Mock _prepare_tree_row to return dummy values
    monkeypatch.setattr(gptscan, "_prepare_tree_row", lambda v: (list(v), ("tag",)))

    return mock_tree, mock_filter_var

def test_update_tree_row_existing_match(mock_gui_globals):
    mock_tree, mock_filter_var = mock_gui_globals
    mock_filter_var.get.return_value = "" # No filter
    mock_tree.exists.return_value = True

    path = "test.py"
    old_values = (path, "50%", "Old Admin", "Old User", "40%", "Old Snippet")
    new_values = (path, "90%", "New Admin", "New User", "80%", "New Snippet")

    gptscan._all_results_cache = [old_values]

    gptscan.update_tree_row("item1", new_values)

    # Verify cache update
    assert gptscan._all_results_cache[0] == new_values

    # Verify tree update
    mock_tree.item.assert_called_once()
    _, kwargs = mock_tree.item.call_args
    assert kwargs["values"][1] == "90%"

def test_update_tree_row_existing_no_match_deletes(mock_gui_globals):
    mock_tree, mock_filter_var = mock_gui_globals
    mock_filter_var.get.return_value = "suspicious" # Filter that doesn't match new values
    mock_tree.exists.return_value = True

    path = "test.py"
    new_values = (path, "10%", "Clean", "Clean", "0%", "print('hi')") # Doesn't match "suspicious"

    gptscan._all_results_cache = [(path, "50%", "Old", "Old", "", "")]

    gptscan.update_tree_row("item1", new_values)

    # Verify tree deletion
    mock_tree.delete.assert_called_once_with("item1")

def test_update_tree_row_hidden_now_matches_refreshes(mock_gui_globals, monkeypatch):
    mock_tree, mock_filter_var = mock_gui_globals
    mock_filter_var.get.return_value = "bad"
    mock_tree.exists.return_value = False # Item not in tree (hidden)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, "_apply_filter", mock_apply_filter)

    path = "bad.py"
    new_values = (path, "95%", "Malicious", "Dangerous", "90%", "eval(x)")

    gptscan._all_results_cache = [(path, "10%", "Clean", "Clean", "", "")]

    gptscan.update_tree_row("item1", new_values)

    # Verify _apply_filter was called to refresh the view
    mock_apply_filter.assert_called_once()

def test_exclude_selected_relpath_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Force Path to resolve relative to tmp_path for extra safety
    from pathlib import Path as RealPath
    monkeypatch.setattr(gptscan, "Path", lambda p: RealPath(tmp_path / p))

    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Mock selection
    mock_tree.selection.return_value = ["item1"]
    def mock_item(item_id, option=None):
        vals = ("C:\\other\\drive\\file.py", "50%", "", "", "", "", "")
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    # Mock askyesno to return True
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args: True)

    # Mock relpath to raise ValueError (simulating different drives on Windows)
    def mock_relpath(path, start):
        raise ValueError("Different drives")
    monkeypatch.setattr(os.path, "relpath", mock_relpath)

    # Mock Config
    monkeypatch.setattr(gptscan.Config, "ignore_patterns", [])
    gptscan._all_results_cache = [("C:\\other\\drive\\file.py", "50%", "", "", "", "")]

    with patch("gptscan._apply_filter"), patch("gptscan.update_status"):
        gptscan.exclude_selected()

    # Verify .gptscanignore contains the absolute path
    ignore_file = Path(".gptscanignore")
    assert ignore_file.exists()
    assert "C:\\other\\drive\\file.py" in ignore_file.read_text()
    assert "C:\\other\\drive\\file.py" in gptscan.Config.ignore_patterns

def test_exclude_selected_json_decode_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from pathlib import Path as RealPath
    monkeypatch.setattr(gptscan, "Path", lambda p: RealPath(tmp_path / p))

    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Mock selection with invalid JSON in hidden column
    mock_tree.selection.return_value = ["item1"]
    def mock_item(item_id, option=None):
        vals = ("wrapped_path.py", "50%", "", "", "", "", "INVALID_JSON")
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args: True)
    monkeypatch.setattr(gptscan.Config, "ignore_patterns", [])

    with patch("gptscan._apply_filter"), patch("gptscan.update_status"):
        gptscan.exclude_selected()

    # It should fallback to the display path
    ignore_file = Path(".gptscanignore")
    assert "wrapped_path.py" in ignore_file.read_text()

def test_copy_as_markdown_json_fallback(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")

    # Selection with invalid JSON in hidden column
    mock_tree.selection.return_value = ["item1"]
    # Treeview values are typically strings in real app
    def mock_item(item_id, option=None):
        vals = ["path.py\nwrapped", "50%", "Admin\nNotes", "User\nNotes", "40%", "Snippet\nCode", "INVALID_JSON"]
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    gptscan.copy_as_markdown()

    # Verify it used the fallback (which replaces newlines with spaces)
    args, _ = mock_tree.clipboard_append.call_args
    markdown_text = args[0]
    assert "path.py wrapped" in markdown_text
    assert "Admin Notes" in markdown_text
