import pytest
from unittest.mock import MagicMock, patch
from gptscan import scan_recently_modified_click

def test_scan_recently_modified_click_no_arg(monkeypatch):
    # Mock simpledialog.askstring to return '1h'
    mock_askstring = MagicMock(return_value='1h')
    monkeypatch.setattr('gptscan.simpledialog.askstring', mock_askstring)

    # Mock button_click to verify it's called
    mock_button_click = MagicMock()
    monkeypatch.setattr('gptscan.button_click', mock_button_click)

    # Mock time.time
    mock_time = MagicMock(return_value=1000000.0)
    monkeypatch.setattr('time.time', mock_time)

    scan_recently_modified_click()

    mock_askstring.assert_called_once()
    mock_button_click.assert_called_once()
    # 1h = 3600s. 1000000 - 3600 = 996400
    assert mock_button_click.call_args[1]['modified_since'] == 996400.0

def test_scan_recently_modified_click_with_arg(monkeypatch):
    # Mock simpledialog.askstring - should NOT be called
    mock_askstring = MagicMock()
    monkeypatch.setattr('gptscan.simpledialog.askstring', mock_askstring)

    # Mock button_click
    mock_button_click = MagicMock()
    monkeypatch.setattr('gptscan.button_click', mock_button_click)

    # Mock time.time
    mock_time = MagicMock(return_value=1000000.0)
    monkeypatch.setattr('time.time', mock_time)

    scan_recently_modified_click("24h")

    mock_askstring.assert_not_called()
    mock_button_click.assert_called_once()
    # 24h = 86400s. 1000000 - 86400 = 913600
    assert mock_button_click.call_args[1]['modified_since'] == 913600.0

def test_scan_recently_modified_click_invalid_duration(monkeypatch):
    # Mock simpledialog.askstring
    mock_askstring = MagicMock(return_value='invalid')
    monkeypatch.setattr('gptscan.simpledialog.askstring', mock_askstring)

    # Mock messagebox.showerror
    mock_showerror = MagicMock()
    monkeypatch.setattr('gptscan.messagebox.showerror', mock_showerror)

    # Mock button_click - should NOT be called
    mock_button_click = MagicMock()
    monkeypatch.setattr('gptscan.button_click', mock_button_click)

    scan_recently_modified_click()

    mock_showerror.assert_called_once()
    mock_button_click.assert_not_called()
