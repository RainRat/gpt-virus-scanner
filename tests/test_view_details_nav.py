import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import tkinter as tk

@pytest.fixture
def mock_gui_elements(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'root', MagicMock())

    # Mock _get_item_raw_values
    mock_values = {
        "item1": ["f1.py", "10%", "A1", "U1", "5%", "code1"],
        "item2": ["f2.py", "20%", "A2", "U2", "15%", "code2"],
        "item3": ["f3.py", "30%", "A3", "U3", "25%", "code3"]
    }
    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda iid: mock_values.get(iid))

    # Mock get_children
    mock_tree.get_children.return_value = ["item1", "item2", "item3"]

    return mock_tree

def test_view_details_navigation(mock_gui_elements, monkeypatch):
    mock_tree = mock_gui_elements
    # Initial selection
    mock_tree.selection.return_value = ["item2"]

    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock scrolledtext
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MagicMock())

    # Mock bind_hover_message
    monkeypatch.setattr(gptscan, 'bind_hover_message', MagicMock())

    # Call view_details for item2 (via selection)
    gptscan.view_details()

    # Verify Toplevel was created for item2
    mock_toplevel.title.assert_called_with("Result Details - f2.py")

    # Call view_details with item3 (navigation)
    gptscan.view_details(item_id="item3")

    # Selection should have been updated to item3
    mock_tree.selection_set.assert_called_with("item3")
    mock_tree.see.assert_called_with("item3")
    # Toplevel should have been destroyed and recreated
    assert mock_toplevel.destroy.called
    mock_toplevel.title.assert_called_with("Result Details - f3.py")

    # Test disabled state for Previous on first item
    gptscan.view_details(item_id="item1")
    # We can't easily check button states here without more mocking,
    # but we can verify it doesn't crash and title is correct.
    mock_toplevel.title.assert_called_with("Result Details - f1.py")
