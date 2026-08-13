import pytest
from unittest.mock import MagicMock
import gptscan
import tkinter as tk

@pytest.fixture
def mock_root_ui_env(monkeypatch):
    mock_tree = MagicMock()
    mock_filter_var = MagicMock()
    mock_apply_filter = MagicMock()
    mock_cancel_scan = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)
    monkeypatch.setattr(gptscan, 'cancel_scan', mock_cancel_scan)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    return {
        'tree': mock_tree,
        'filter_var': mock_filter_var,
        'apply_filter': mock_apply_filter,
        'cancel_scan': mock_cancel_scan
    }

def test_on_root_escape_scan_active(mock_root_ui_env, monkeypatch):
    """Test that when a scan is active, pressing Escape globally stops the scan."""
    mock_event = MagicMock()
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)

    res = gptscan.on_root_escape()

    mock_root_ui_env['cancel_scan'].assert_called_once()
    assert res == "break"
    # Filter should not be touched since scan was active
    mock_root_ui_env['filter_var'].set.assert_not_called()

def test_on_root_escape_scan_inactive_with_filter(mock_root_ui_env, monkeypatch):
    """Test that when no scan is active but a filter query exists, pressing Escape clears it and focuses the results."""
    mock_root_ui_env['filter_var'].get.return_value = "suspicious_term"
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    res = gptscan.on_root_escape()

    # Cancel scan should NOT be called
    mock_root_ui_env['cancel_scan'].assert_not_called()
    # Filter should be cleared
    mock_root_ui_env['filter_var'].set.assert_called_once_with("")
    # Results should be refreshed
    mock_root_ui_env['apply_filter'].assert_called_once()
    # Focus should shift back to the tree
    mock_root_ui_env['tree'].focus_set.assert_called_once()
    assert res == "break"

def test_on_root_escape_scan_inactive_no_filter(mock_root_ui_env, monkeypatch):
    """Test that when no scan is active and no filter query exists, pressing Escape does nothing."""
    mock_root_ui_env['filter_var'].get.return_value = "   "  # Whitespace only
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    res = gptscan.on_root_escape()

    mock_root_ui_env['cancel_scan'].assert_not_called()
    mock_root_ui_env['filter_var'].set.assert_not_called()
    mock_root_ui_env['apply_filter'].assert_not_called()
    mock_root_ui_env['tree'].focus_set.assert_not_called()
    assert res == ""

def test_on_root_escape_handles_none_filter_or_tree(monkeypatch):
    """Test that on_root_escape behaves gracefully if filter_var or tree is None."""
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    monkeypatch.setattr(gptscan, 'filter_var', None)
    monkeypatch.setattr(gptscan, 'tree', None)

    res = gptscan.on_root_escape()

    assert res == ""
