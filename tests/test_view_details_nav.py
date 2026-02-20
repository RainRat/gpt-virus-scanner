import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import tkinter as tk
import os

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    # Mock tree.get_children
    mock_tree.get_children.return_value = ["item1", "item2", "item3"]

    # Mock data for items
    # Format in Treeview: (path, own_conf, admin, user, gpt_conf, snippet, orig_json)
    mock_data = {
        "item1": ["file1.py", "10%", "", "", "", "snippet1", json.dumps(["file1.py", "10%", "", "", "", "snippet1"])],
        "item2": ["file2.py", "20%", "Admin", "User", "90%", "snippet2", json.dumps(["file2.py", "20%", "Admin", "User", "90%", "snippet2"])],
        "item3": ["file3.py", "30%", "", "", "", "snippet3", json.dumps(["file3.py", "30%", "", "", "", "snippet3"])]
    }

    def mock_item_func(item_id, option=None):
        vals = mock_data.get(item_id, [])
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree.exists.side_effect = lambda iid: iid in mock_data

    return mock_tree

def test_view_details_navigation(mock_tree, monkeypatch):
    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock ScrolledText
    mock_scrolledtext = MagicMock()
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MagicMock(return_value=mock_scrolledtext))

    # Capture commands of buttons
    captured_commands = {}
    original_button = gptscan.ttk.Button
    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured_commands[text] = kwargs.get('command')
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    # Call view_details starting with item2
    gptscan.view_details(item_id="item2")

    # Verify initial state
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file2.py"
    assert "< Previous" in captured_commands
    assert "Next >" in captured_commands

    # Test "Next >" button
    captured_commands["Next >"]()
    # Verify it updated selection and title
    mock_tree.selection_set.assert_called_with("item3")
    mock_tree.see.assert_called_with("item3")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file3.py"

    # Test "< Previous" button
    captured_commands["< Previous"]()
    # Verify it updated selection and title back to item2
    mock_tree.selection_set.assert_called_with("item2")
    mock_tree.see.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file2.py"

    # Test Previous again to item1
    captured_commands["< Previous"]()
    mock_tree.selection_set.assert_called_with("item1")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file1.py"

def test_view_details_keyboard_bindings(mock_tree, monkeypatch):
    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MagicMock())

    # Capture bindings
    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item2")

    assert '<Left>' in captured_bindings
    assert '<Right>' in captured_bindings

    # Trigger Right arrow
    captured_bindings['<Right>'](None)
    mock_tree.selection_set.assert_called_with("item3")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file3.py"

    # Trigger Left arrow
    captured_bindings['<Left>'](None)
    mock_tree.selection_set.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result Details - file2.py"
