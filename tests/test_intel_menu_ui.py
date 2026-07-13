import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk
import json

@pytest.fixture
def mock_gui(monkeypatch):
    mock_tree = MagicMock()
    mock_intel_button = MagicMock()
    mock_intel_menu = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'intel_button', mock_intel_button)
    monkeypatch.setattr(gptscan, 'intel_menu', mock_intel_menu)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    return mock_tree, mock_intel_button, mock_intel_menu

def test_intel_menu_states_no_selection(mock_gui):
    """Test that Intel menu and button are disabled when nothing is selected."""
    mock_tree, mock_intel_button, mock_intel_menu = mock_gui
    mock_tree.selection.return_value = []

    gptscan.update_button_states()

    mock_intel_button.config.assert_called_with(state="disabled")
    mock_intel_menu.entryconfig.assert_any_call("Check on VirusTotal", state="disabled")
    mock_intel_menu.entryconfig.assert_any_call("View Online", state="disabled")

def test_intel_menu_states_with_selection(mock_gui, monkeypatch):
    """Test that Intel menu and button are enabled when a file is selected."""
    mock_tree, mock_intel_button, mock_intel_menu = mock_gui
    mock_tree.selection.return_value = ["item1"]

    # Mock _get_item_raw_values to return a standard local path
    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda item_id: ["test.py", "50%"])

    gptscan.update_button_states()

    mock_intel_button.config.assert_called_with(state="normal")
    mock_intel_menu.entryconfig.assert_any_call("Check on VirusTotal", state="normal")
    mock_intel_menu.entryconfig.assert_any_call("View Online", state="normal")

def test_intel_menu_states_virtual_path(mock_gui, monkeypatch):
    """Test that View Online is disabled for non-URL virtual paths (like [Clipboard])."""
    mock_tree, mock_intel_button, mock_intel_menu = mock_gui
    mock_tree.selection.return_value = ["item1"]

    # Mock _get_item_raw_values to return a virtual path
    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda item_id: ["[Clipboard]", "0%"])

    gptscan.update_button_states()

    mock_intel_button.config.assert_called_with(state="normal")
    mock_intel_menu.entryconfig.assert_any_call("Check on VirusTotal", state="normal")
    mock_intel_menu.entryconfig.assert_any_call("View Online", state="disabled")

def test_intel_button_disabled_during_scan(mock_gui, monkeypatch):
    """Test that the Intel button is disabled when a scan is active."""
    mock_tree, mock_intel_button, mock_intel_menu = mock_gui

    # Setup scan state widgets
    monkeypatch_config = [
        'textbox', 'clear_target_btn', 'browse_button',
        'git_checkbox', 'deep_checkbox', 'scan_all_checkbox', 'dry_checkbox',
        'gpt_checkbox', 'provider_combo', 'model_combo', 'api_entry', 'show_key_btn',
        'copy_cmd_button', 'scan_button'
    ]
    for name in monkeypatch_config:
        monkeypatch.setattr(gptscan, name, MagicMock())

    gptscan.set_scanning_state(True)

    mock_intel_button.config.assert_called_with(state="disabled")
