import tkinter as tk
from tkinter import ttk
import tkinter.font
from unittest.mock import MagicMock
import gptscan

def test_motion_handler_updates_children(monkeypatch):
    # Setup a mock tree
    mock_tree = MagicMock()

    # Mock tree['columns']
    mock_tree.__getitem__.side_effect = lambda key: ("col1", "col2") if key == "columns" else MagicMock()

    # Mock tree.column(cid)['width']
    mock_tree.column.return_value = {'width': 100}

    # Mock tree.get_children()
    mock_tree.get_children.return_value = ["iid1"]

    # Mock tree.item(iid)['values']
    # tree.item(iid) returns a dict, tree.item(iid, values=...) updates it.
    def mock_item(iid, values=None):
        if values is None:
            return {'values': ["long text that should wrap", "short"]}
        return None

    mock_tree.item.side_effect = mock_item

    # Mock font measure
    monkeypatch.setattr(tkinter.font.Font, 'measure', lambda self, text: len(text) * 10)

    # Mock adjust_newlines to return a predictable value
    monkeypatch.setattr(gptscan, "adjust_newlines", lambda text, width, measure=None: f"wrapped_{text}")

    # Call motion_handler with event=None
    gptscan.motion_handler(mock_tree, None)

    # Verify tree.item(iid, values=...) was called with wrapped values
    mock_tree.item.assert_called_with("iid1", values=["wrapped_long text that should wrap", "wrapped_short"])

def test_motion_handler_ignores_non_separator_events():
    mock_tree = MagicMock()
    mock_tree.identify_region.return_value = "cell"

    mock_event = MagicMock()
    mock_event.x = 10
    mock_event.y = 10

    gptscan.motion_handler(mock_tree, mock_event)

    # Should not have called get_children
    mock_tree.get_children.assert_not_called()

def test_motion_handler_triggers_on_separator(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.identify_region.return_value = "separator"
    mock_tree.get_children.return_value = []

    mock_event = MagicMock()
    mock_event.x = 10
    mock_event.y = 10

    # Mock font measure to avoid needing a real display
    monkeypatch.setattr(tkinter.font.Font, 'measure', lambda self, text: len(text) * 10)

    gptscan.motion_handler(mock_tree, mock_event)

    # Should have called get_children because it was a separator
    mock_tree.get_children.assert_called_once()
