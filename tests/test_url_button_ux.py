import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_browse_url_click_updates_textbox(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    mock_scan_button = MagicMock()
    monkeypatch.setattr(gptscan, 'scan_button', mock_scan_button, raising=False)

    # Mock simpledialog.askstring to return a URL
    url = "https://example.com/script.sh"
    monkeypatch.setattr(gptscan.simpledialog, 'askstring', lambda title, prompt: url)

    # Call function
    gptscan.browse_url_click()

    # Assert
    # _set_scan_target is called internally which does delete and insert
    mock_textbox.delete.assert_called_with(0, gptscan.tk.END)
    mock_textbox.insert.assert_called_with(0, url)
    mock_scan_button.focus_set.assert_called_once()

def test_browse_url_click_cancels(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    # Mock simpledialog.askstring to return None (cancellation)
    monkeypatch.setattr(gptscan.simpledialog, 'askstring', lambda title, prompt: None)

    # Call function
    gptscan.browse_url_click()

    # Assert
    mock_textbox.delete.assert_not_called()
    mock_textbox.insert.assert_not_called()

def test_set_scanning_state_disables_url_button(monkeypatch):
    # Setup mocks
    mock_url_btn = MagicMock()
    monkeypatch.setattr(gptscan, 'select_url_btn', mock_url_btn, raising=False)

    # We need to mock other widgets used in set_scanning_state to avoid errors
    monkeypatch.setattr(gptscan, 'scan_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'textbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'select_file_btn', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'select_dir_btn', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'select_clipboard_btn', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'git_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'deep_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'scan_all_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'dry_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'gpt_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'provider_combo', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'model_combo', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'copy_cmd_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'all_checkbox', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'threshold_spin', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'view_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'rescan_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'open_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'analyze_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'exclude_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'reveal_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'import_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'export_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'clear_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'toggle_ai_controls', MagicMock())
    monkeypatch.setattr(gptscan, 'update_button_states', MagicMock())

    # Test scanning=True
    gptscan.set_scanning_state(True)
    mock_url_btn.config.assert_called_with(state="disabled")

    # Test scanning=False
    gptscan.set_scanning_state(False)
    mock_url_btn.config.assert_called_with(state="normal")
