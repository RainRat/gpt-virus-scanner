import pytest
import json
import tkinter as tk
from unittest.mock import MagicMock, patch
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details

def test_view_details_copy_code(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a result
    raw = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    # Find the Copy Code button and command
    assert "btn_Copy Code" in captured
    btn_mock, copy_code_cmd = captured["btn_Copy Code"]

    # Execute the command
    from gptscan import root as mock_root
    copy_code_cmd()

    # Verify clipboard and status bar feedback
    mock_root.clipboard_clear.assert_called()
    mock_root.clipboard_append.assert_called_with("print('hello')")
    # Verified feedback via status bar (it's the first label created in view_details)
    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Code copied to clipboard."

def test_view_details_copy_path_moved(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py")

    # Verify button exists with new text
    assert "btn_Copy" in captured
    btn_mock, copy_path_cmd = captured["btn_Copy"]

    from gptscan import root as mock_root
    copy_path_cmd()
    mock_root.clipboard_append.assert_called_with("test.py")

def test_view_details_shortcuts(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet_content")

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item1")

    assert '<Control-s>' in captured_bindings
    assert '<Command-s>' in captured_bindings

    from gptscan import root as mock_root
    captured_bindings['<Control-s>'](None)
    mock_root.clipboard_append.assert_called_with("snippet_content")
