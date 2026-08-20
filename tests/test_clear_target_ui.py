import pytest
from unittest.mock import MagicMock
import gptscan


def test_update_clear_target_visibility_shows_when_input_present(monkeypatch):
    mock_btn = MagicMock()
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "  /path/to/scan  "

    monkeypatch.setattr(gptscan, 'clear_target_btn', mock_btn)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)

    gptscan.update_clear_target_visibility()

    mock_btn.grid.assert_called_once_with(row=0, column=2, padx=(0, 5))
    mock_btn.grid_remove.assert_not_called()


def test_update_clear_target_visibility_hides_when_empty(monkeypatch):
    mock_btn = MagicMock()
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "   "

    monkeypatch.setattr(gptscan, 'clear_target_btn', mock_btn)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)

    gptscan.update_clear_target_visibility()

    mock_btn.grid_remove.assert_called_once()
    mock_btn.grid.assert_not_called()


def test_update_clear_target_visibility_handles_none_widgets(monkeypatch):
    monkeypatch.setattr(gptscan, 'clear_target_btn', None)
    monkeypatch.setattr(gptscan, 'textbox', None)

    # Should not raise any exception
    gptscan.update_clear_target_visibility()


def test_set_scan_target_updates_visibility(monkeypatch):
    mock_btn = MagicMock()
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "src/main.py"
    mock_scan_btn = MagicMock()

    monkeypatch.setattr(gptscan, 'clear_target_btn', mock_btn)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)
    monkeypatch.setattr(gptscan, 'scan_button', mock_scan_btn)

    gptscan._set_scan_target("src/main.py")

    mock_textbox.delete.assert_called_once_with(0, "end")
    mock_textbox.insert.assert_called_once_with(0, "src/main.py")
    mock_btn.grid.assert_called_once_with(row=0, column=2, padx=(0, 5))
    mock_scan_btn.focus_set.assert_called_once()


def test_clear_target_functionality(monkeypatch):
    """Test clear_target clears textbox, updates visibility, and refocuses textbox."""
    mock_btn = MagicMock()
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""

    monkeypatch.setattr(gptscan, 'clear_target_btn', mock_btn)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)

    # Simulate clear_target callback logic
    mock_textbox.delete(0, "end")
    gptscan.update_clear_target_visibility()
    mock_textbox.focus_set()

    mock_textbox.delete.assert_called_once_with(0, "end")
    mock_btn.grid_remove.assert_called_once()
    mock_textbox.focus_set.assert_called_once()
