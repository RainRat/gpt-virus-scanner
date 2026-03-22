import pytest
from unittest.mock import MagicMock, patch
import gptscan
import threading

@pytest.fixture
def mock_gui(monkeypatch):
    mock_root = MagicMock()
    mock_textbox = MagicMock()
    mock_messagebox = MagicMock()
    mock_git_var = MagicMock()
    mock_dry_var = MagicMock()
    mock_deep_var = MagicMock()
    mock_all_var = MagicMock()
    mock_gpt_var = MagicMock()

    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)
    monkeypatch.setattr(gptscan, 'messagebox', mock_messagebox)
    monkeypatch.setattr(gptscan, 'git_var', mock_git_var)
    monkeypatch.setattr(gptscan, 'dry_var', mock_dry_var)
    monkeypatch.setattr(gptscan, 'deep_var', mock_deep_var)
    monkeypatch.setattr(gptscan, 'all_var', mock_all_var)
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var)

    # Mocking finish_scan_state and update_status to avoid errors
    monkeypatch.setattr(gptscan, 'set_scanning_state', MagicMock())
    monkeypatch.setattr(gptscan, 'update_status', MagicMock())
    monkeypatch.setattr(gptscan, 'clear_results', MagicMock())

    return {
        'root': mock_root,
        'textbox': mock_textbox,
        'messagebox': mock_messagebox,
        'git_var': mock_git_var,
        'dry_var': mock_dry_var
    }

def test_button_click_with_empty_path_and_snippets(mock_gui, monkeypatch):
    """Verify that button_click proceeds if snippets are provided, even if scan_path is empty."""
    mock_gui['textbox'].get.return_value = ""
    mock_gui['git_var'].get.return_value = False
    mock_gui['dry_var'].get.return_value = True # Avoid model check for simplicity

    # Mock Thread to avoid actual execution
    mock_thread = MagicMock()
    monkeypatch.setattr(threading, 'Thread', mock_thread)

    extra_snippets = [("[Clipboard]", b"print('hello')")]
    gptscan.button_click(extra_snippets=extra_snippets)

    # Should NOT have shown an error
    mock_gui['messagebox'].showerror.assert_not_called()

    # Should HAVE started a scan thread
    mock_thread.assert_called_once()
    args, kwargs = mock_thread.call_args
    scan_args = kwargs.get('args') or args[1]

    # scan_targets (first arg of scan_args) should be an empty list, not an empty string or current dir
    assert scan_args[0] == []
    # extra_snippets (9th arg) should be present
    assert scan_args[8] == extra_snippets

def test_button_click_with_empty_path_and_no_snippets(mock_gui, monkeypatch):
    """Verify that button_click still shows error if both path and snippets are missing."""
    mock_gui['textbox'].get.return_value = ""

    gptscan.button_click(extra_snippets=None)

    # Should HAVE shown an error
    mock_gui['messagebox'].showerror.assert_called_with("Missing Selection", "Please select a file or folder to scan.")
