import pytest
from unittest.mock import MagicMock, patch
import gptscan
import os
import sys

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    # Standard mock for tree.item that handles both dict return and "values" option
    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, ())
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree._item_values = {}
    return mock_tree

import json

def test_copy_path(mock_tree):
    mock_tree.selection.return_value = ["item1"]
    # Path is the first column. Wrapped for display.
    # Hidden orig_json stores the raw values.
    raw_values = ["some/path/to/file.py", "90%", "Admin", "User", "80%", "print('hi')"]
    mock_tree._item_values["item1"] = ("some/path/to/\nfile.py", "90%", "Admin", "User", "80%", "print('hi')", json.dumps(raw_values))

    gptscan.copy_path()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_with("some/path/to/file.py")

def test_copy_snippet(mock_tree):
    mock_tree.selection.return_value = ["item1"]
    # Snippet is the last column. Wrapped for display.
    raw_values = ["some/path", "90%", "Admin", "User", "80%", "print('wrapped\nsnippet')"]
    mock_tree._item_values["item1"] = ("some/path", "90%", "Admin", "User", "80%", "print('wrapped\nsnippet')", json.dumps(raw_values))

    gptscan.copy_snippet()

    mock_tree.clipboard_clear.assert_called_once()
    # Expecting it to preserve the newline from the original data
    mock_tree.clipboard_append.assert_called_with("print('wrapped\nsnippet')")

def test_show_in_folder_windows(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("C:\\path\\to\\file.py",)

    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'win32'):
        mock_run = MagicMock()
        monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)
        monkeypatch.setattr(os.path, 'normpath', lambda p: p)

        gptscan.show_in_folder()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "explorer" in args
        assert "/select," in args

def test_show_in_folder_mac(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("/Users/test/file.py",)

    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'darwin'):
        mock_run = MagicMock()
        monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)

        gptscan.show_in_folder()

        mock_run.assert_called_once_with(["open", "-R", "/Users/test/file.py"])

def test_show_in_folder_linux(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("/home/user/file.py",)

    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'linux'):
        mock_run = MagicMock()
        monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)

        gptscan.show_in_folder()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "xdg-open" in args
        assert "/home/user" in args

def test_open_file_windows(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("C:\\path\\to\\file.py",)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'win32'):
        mock_startfile = MagicMock()
        monkeypatch.setattr(gptscan.os, 'startfile', mock_startfile, raising=False)
        gptscan.open_file()
        mock_startfile.assert_called_once_with("C:\\path\\to\\file.py")

def test_open_file_mac(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("/Users/test/file.py",)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'darwin'):
        mock_run = MagicMock()
        monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)
        gptscan.open_file()
        mock_run.assert_called_once_with(["open", "/Users/test/file.py"])

def test_open_file_linux(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("/home/user/file.py",)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    with patch('sys.platform', 'linux'):
        mock_run = MagicMock()
        monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)
        gptscan.open_file()
        mock_run.assert_called_once_with(["xdg-open", "/home/user/file.py"])

def test_open_file_not_found(mock_tree, monkeypatch):
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("ghost.py",)
    monkeypatch.setattr(os.path, 'exists', lambda p: False)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    gptscan.open_file()
    mock_msgbox.showwarning.assert_called_with("File Not Found", "The file 'ghost.py' could not be located.")

def test_show_context_menu_already_selected(mock_tree, monkeypatch):
    mock_event = MagicMock()
    mock_event.y = 100
    mock_event.x_root = 200
    mock_event.y_root = 300

    mock_tree.identify_row.return_value = "item1"
    mock_tree.selection.return_value = ["item1"]

    mock_menu = MagicMock()
    monkeypatch.setattr(gptscan, 'context_menu', mock_menu, raising=False)

    gptscan.show_context_menu(mock_event)

    mock_tree.selection_set.assert_not_called()
    mock_menu.post.assert_called_with(200, 300)

def test_show_context_menu_new_selection(mock_tree, monkeypatch):
    mock_event = MagicMock()
    mock_event.y = 100
    mock_event.x_root = 200
    mock_event.y_root = 300

    mock_tree.identify_row.return_value = "item1"
    mock_tree.selection.side_effect = [[], ["item1"]]

    mock_menu = MagicMock()
    monkeypatch.setattr(gptscan, 'context_menu', mock_menu, raising=False)

    gptscan.show_context_menu(mock_event)

    mock_tree.selection_set.assert_called_with("item1")
    mock_menu.post.assert_called_with(200, 300)

def test_show_context_menu_no_item_no_selection(mock_tree, monkeypatch):
    mock_event = MagicMock()
    mock_event.y = 100
    mock_tree.identify_row.return_value = ""
    mock_tree.selection.return_value = []

    mock_menu = MagicMock()
    monkeypatch.setattr(gptscan, 'context_menu', mock_menu, raising=False)

    gptscan.show_context_menu(mock_event)

    mock_tree.selection_set.assert_not_called()
    mock_menu.post.assert_not_called()

def test_copy_as_markdown(mock_tree):
    mock_tree.selection.return_value = ["item1"]
    raw_values = ["test.py", "90%", "Admin", "User", "80%", "print('hi')"]
    mock_tree._item_values["item1"] = ("test.py", "90%", "Admin", "User", "80%", "print('hi')", json.dumps(raw_values))
    # Mock tree["columns"]
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")

    gptscan.copy_as_markdown()

    mock_tree.clipboard_clear.assert_called_once()
    args, _ = mock_tree.clipboard_append.call_args
    md = args[0]
    assert "| test.py | 80% | **Admin:** Admin<br>**User:** User | `print('hi')` |" in md

def test_select_all_items(mock_tree):
    mock_tree.get_children.return_value = ("item1", "item2", "item3")

    gptscan.select_all_items()

    mock_tree.selection_set.assert_called_with(("item1", "item2", "item3"))
