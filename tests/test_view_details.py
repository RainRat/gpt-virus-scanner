import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import sys
import tkinter as tk

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    # Mock get_children for navigation logic
    mock_tree.get_children.return_value = ["item1"]

    # Mock _get_selected_row_values dependencies
    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, ())
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree._item_values = {}
    return mock_tree

def test_view_details_open_window(mock_tree, monkeypatch):
    # Setup mock selection
    mock_tree.selection.return_value = ["item1"]
    raw_values = ["test.py", "90%", "Admin Note", "User Note", "85%", "print('hello')"]
    mock_tree._item_values["item1"] = ("test.py", "90%", "Admin Note", "User Note", "85%", "print('hello')", json.dumps(raw_values))

    # Mock Toplevel and other Tkinter widgets
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock scrolledtext
    mock_scrolledtext = MagicMock()
    mock_st_class = MagicMock(return_value=mock_scrolledtext)

    # Patch the already imported module in gptscan
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', mock_st_class)

    gptscan.view_details()

    # Verify Toplevel was created with correct title
    gptscan.tk.Toplevel.assert_called_once()
    mock_toplevel.title.assert_called_with("Result 1 of 1 - test.py")

    # Verify data was inserted into ScrolledText widgets (we expect 3: admin, user, snippet)
    # Actually if both admin and user are present, there are 3 ScrolledText calls.
    assert mock_st_class.call_count == 3

    # Check that the data was inserted
    calls = mock_scrolledtext.insert.call_args_list
    contents = [call[0][1] for call in calls]
    assert "Admin Note" in contents
    assert "User Note" in contents
    assert "print('hello')" in contents

def test_view_details_no_selection(mock_tree, monkeypatch):
    mock_tree.selection.return_value = []

    mock_toplevel_init = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', mock_toplevel_init)

    gptscan.view_details()

    mock_toplevel_init.assert_not_called()
