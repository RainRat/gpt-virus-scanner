import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as scrolledtext
from unittest.mock import MagicMock
import gptscan

class MockEntryClass:
    pass

class MockTextClass:
    pass

class MockButtonClass:
    pass

def test_clear_results(monkeypatch):
    # Mock global variables
    mock_tree = MagicMock()
    mock_progress_bar = MagicMock()
    mock_status_label = MagicMock()
    mock_root = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Pre-checks
    mock_tree.get_children.return_value = ('item1', 'item2')

    # Call clear_results without event
    res = gptscan.clear_results()

    # Assertions
    assert res == "break"
    mock_tree.delete.assert_called_with('item1', 'item2')
    # For progress_bar['value'] = 0
    mock_progress_bar.__setitem__.assert_called_with('value', 0)
    # update_status calls status_label.config and root.update_idletasks
    mock_status_label.config.assert_called_with(text="Ready")
    mock_root.update_idletasks.assert_called()


def test_clear_results_event_with_focused_entry(monkeypatch):
    # Patch classes in both modules to be distinct mock classes (not MagicMock)
    monkeypatch.setattr(gptscan.ttk, 'Entry', MockEntryClass)
    monkeypatch.setattr(gptscan.tk, 'Entry', MockEntryClass)
    monkeypatch.setattr(gptscan.tk, 'Text', MockTextClass)
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MockTextClass)

    mock_tree = MagicMock()
    mock_progress_bar = MagicMock()
    mock_status_label = MagicMock()
    mock_root = MagicMock()

    # focused is a MockEntryClass instance
    mock_entry = MockEntryClass()
    mock_root.focus_get.return_value = mock_entry

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    mock_event = MagicMock()

    # Call clear_results with an event while focused on MockEntryClass
    res = gptscan.clear_results(mock_event)

    # Assertions: should return None and not clear results
    assert res is None
    mock_tree.delete.assert_not_called()


def test_clear_results_event_with_non_entry_focused(monkeypatch):
    # Patch classes in both modules to be distinct mock classes (not MagicMock)
    monkeypatch.setattr(gptscan.ttk, 'Entry', MockEntryClass)
    monkeypatch.setattr(gptscan.tk, 'Entry', MockEntryClass)
    monkeypatch.setattr(gptscan.tk, 'Text', MockTextClass)
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MockTextClass)

    mock_tree = MagicMock()
    mock_progress_bar = MagicMock()
    mock_status_label = MagicMock()
    mock_root = MagicMock()

    # focused is a MockButtonClass instance, which is not an Entry or Text
    mock_btn = MockButtonClass()
    mock_root.focus_get.return_value = mock_btn

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    mock_event = MagicMock()

    # Call clear_results with an event while focused on MockButtonClass
    res = gptscan.clear_results(mock_event)

    # Assertions: should return "break" and clear results
    assert res == "break"
    mock_tree.delete.assert_called()
