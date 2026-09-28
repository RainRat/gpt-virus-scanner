import json
from unittest.mock import MagicMock
import pytest
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details


def test_is_input_focused_returns_true_for_input_widgets(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    raw2 = ["file2.py", "20%", "", "", "", "snippet2", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "", "", "", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    gptscan.view_details(item_id="item1")

    prev_btn, prev_cmd = captured["btn_< Previous"]
    next_btn, next_cmd = captured["btn_Next >"]

    for widget_class in ("Text", "Entry", "TEntry"):
        mock_focused = MagicMock()
        mock_focused.winfo_class.return_value = widget_class
        mock_toplevel.focus_get.return_value = mock_focused

        mock_tree.selection_set.reset_mock()

        prev_cmd()
        next_cmd()

        mock_tree.selection_set.assert_not_called()


def test_is_input_focused_returns_false_for_non_input_and_none(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    raw2 = ["file2.py", "20%", "", "", "", "snippet2", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "", "", "", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    gptscan.view_details(item_id="item1")

    next_btn, next_cmd = captured["btn_Next >"]

    mock_focused = MagicMock()
    mock_focused.winfo_class.return_value = "Button"
    mock_toplevel.focus_get.return_value = mock_focused

    mock_tree.selection_set.reset_mock()
    next_cmd()
    mock_tree.selection_set.assert_called_with("item2")

    mock_toplevel.focus_get.return_value = None
    gptscan.view_details(item_id="item1")
    next_btn2, next_cmd2 = captured["btn_Next >"]
    mock_tree.selection_set.reset_mock()
    next_cmd2()
    mock_tree.selection_set.assert_called_with("item2")


def test_is_input_focused_exception_handling(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    raw2 = ["file2.py", "20%", "", "", "", "snippet2", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "", "", "", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    gptscan.view_details(item_id="item1")

    next_btn, next_cmd = captured["btn_Next >"]

    mock_focused = MagicMock()
    mock_focused.winfo_class.side_effect = RuntimeError("Widget destroyed")
    mock_toplevel.focus_get.return_value = mock_focused

    mock_tree.selection_set.reset_mock()
    next_cmd()
    mock_tree.selection_set.assert_called_with("item2")
