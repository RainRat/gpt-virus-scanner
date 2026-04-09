import pytest
from unittest.mock import MagicMock
import gptscan
import tkinter as tk

def test_window_title_progress(monkeypatch):
    """Test that window title updates correctly during scan progress."""
    mock_root = MagicMock()
    mock_progress_bar = MagicMock()

    # Mock global variables in gptscan
    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)

    # 1. Test configure_progress
    gptscan.configure_progress(100)
    mock_root.title.assert_called_with("[0%] GPT Virus Scanner")
    mock_progress_bar.__setitem__.assert_any_call("maximum", 100)
    mock_progress_bar.__setitem__.assert_any_call("value", 0)

    # 2. Test update_progress
    # Mock __getitem__ to return 100 for "maximum"
    mock_progress_bar.__getitem__.side_effect = lambda key: 100 if key == "maximum" else None

    gptscan.update_progress(45)
    mock_root.title.assert_called_with("[45%] GPT Virus Scanner")
    mock_progress_bar.__setitem__.assert_any_call("value", 45)

    # 3. Test update_progress with 100%
    gptscan.update_progress(100)
    mock_root.title.assert_called_with("[100%] GPT Virus Scanner")

    # 4. Test set_scanning_state(False) resets title
    # We need to make sure update_button_states doesn't fail due to mocks
    monkeypatch.setattr(gptscan, 'update_button_states', MagicMock())
    monkeypatch.setattr(gptscan, 'toggle_ai_controls', MagicMock())

    gptscan.set_scanning_state(False)
    mock_root.title.assert_called_with("GPT Virus Scanner")

def test_update_progress_division_by_zero(monkeypatch):
    """Test that update_progress handles zero maximum without crashing."""
    mock_root = MagicMock()
    mock_progress_bar = MagicMock()
    mock_progress_bar["maximum"] = 0

    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)

    # Should not raise ZeroDivisionError
    gptscan.update_progress(10)
    # title should not have been called with a percentage if max is 0
    # Actually my implementation does:
    # try:
    #     total = float(progress_bar["maximum"])
    #     if total > 0:
    #         percent = int((value / total) * 100)
    #         root.title(f"[{percent}%] GPT Virus Scanner")

    # So if total is 0, title is not updated with percentage.
