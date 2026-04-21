import pytest
import json
import tkinter as tk
from unittest.mock import MagicMock, patch
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details

def test_view_details_copy_json(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a result
    raw = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    # We need to mock _get_tree_results_as_dicts or ensure it works with our mock tree
    # Actually _get_tree_results_as_dicts uses tree.item(iid, 'values') and other tree methods.
    # Our mock_tree in mock_view_details_env is already somewhat set up.

    # Let's mock _get_tree_results_as_dicts to be sure what it returns
    mock_json_result = [{"path": "test.py", "threat_level": "90%"}]
    with patch("gptscan._get_tree_results_as_dicts", return_value=mock_json_result):
        gptscan.view_details(item_id="item1")

        # Find the Copy as JSON menu item and command
        assert "menu_Copy as JSON" in captured
        copy_json_cmd = captured["menu_Copy as JSON"]

        # Execute the command
        from gptscan import root as mock_root
        copy_json_cmd()

        # Verify clipboard and status bar feedback
        mock_root.clipboard_clear.assert_called()
        # It should call json.dumps on the first element of the list
        expected_json = json.dumps(mock_json_result[0], indent=2)
        mock_root.clipboard_append.assert_called_with(expected_json)

        # Verify feedback via status bar (it's the first label created in view_details)
        status_bar = captured['labels'][0]
        assert status_bar.config_data.get('text') == "Result copied as JSON."

def test_view_details_json_shortcuts(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    mock_json_result = [{"path": "test.py", "threat_level": "90%"}]
    with patch("gptscan._get_tree_results_as_dicts", return_value=mock_json_result):
        gptscan.view_details(item_id="item1")

        assert '<Control-j>' in captured_bindings
        assert '<Command-j>' in captured_bindings

        from gptscan import root as mock_root
        captured_bindings['<Control-j>'](None)

        expected_json = json.dumps(mock_json_result[0], indent=2)
        mock_root.clipboard_append.assert_called_with(expected_json)

        status_bar = captured['labels'][0]
        assert status_bar.config_data.get('text') == "Result copied as JSON."
