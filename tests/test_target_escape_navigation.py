import pytest
from unittest.mock import MagicMock
import gptscan

@pytest.fixture
def mock_target_escape_env(monkeypatch):
    mock_textbox = MagicMock()
    mock_cancel_scan = MagicMock()
    mock_on_root_escape = MagicMock()
    mock_update_clear_btn = MagicMock()

    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)
    monkeypatch.setattr(gptscan, 'cancel_scan', mock_cancel_scan)
    monkeypatch.setattr(gptscan, 'on_root_escape', mock_on_root_escape)
    monkeypatch.setattr(gptscan, 'update_clear_target_visibility', mock_update_clear_btn)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    return {
        'textbox': mock_textbox,
        'cancel_scan': mock_cancel_scan,
        'on_root_escape': mock_on_root_escape,
        'update_clear_btn': mock_update_clear_btn
    }

def test_on_target_escape_scan_active(mock_target_escape_env, monkeypatch):
    """Test that pressing Escape in target textbox while scan is active cancels the scan."""
    mock_event = MagicMock()
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)

    res = gptscan.on_target_escape()

    mock_target_escape_env['cancel_scan'].assert_called_once()
    assert res == "break"

def test_on_target_escape_clears_target_text(mock_target_escape_env):
    """Test that pressing Escape in target textbox with text clears the entry and keeps focus."""
    mock_target_escape_env['textbox'].get.return_value = "./my_scripts"

    res = gptscan.on_target_escape()

    mock_target_escape_env['textbox'].delete.assert_called_once_with(0, gptscan.tk.END)
    mock_target_escape_env['update_clear_btn'].assert_called_once()
    mock_target_escape_env['textbox'].focus_set.assert_called_once()
    mock_target_escape_env['on_root_escape'].assert_not_called()
    assert res == "break"

def test_on_target_escape_empty_target_delegates_to_root_escape(mock_target_escape_env):
    """Test that pressing Escape in empty target textbox delegates to on_root_escape."""
    mock_target_escape_env['textbox'].get.return_value = "   "
    mock_target_escape_env['on_root_escape'].return_value = "break"

    res = gptscan.on_target_escape()

    mock_target_escape_env['textbox'].delete.assert_not_called()
    mock_target_escape_env['on_root_escape'].assert_called_once()
    assert res == "break"

def test_on_target_escape_handles_none_textbox(monkeypatch):
    """Test that on_target_escape gracefully handles None textbox."""
    monkeypatch.setattr(gptscan, 'textbox', None)
    mock_on_root_escape = MagicMock(return_value="")
    monkeypatch.setattr(gptscan, 'on_root_escape', mock_on_root_escape)

    res = gptscan.on_target_escape()

    mock_on_root_escape.assert_called_once()
    assert res == ""
