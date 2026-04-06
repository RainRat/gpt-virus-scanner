import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_configure_progress_updates_title(monkeypatch):
    mock_root = MagicMock()
    mock_bar = MagicMock()
    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "progress_bar", mock_bar)

    gptscan.configure_progress(100)

    mock_root.title.assert_called_with("[0%] GPT Virus Scanner")

def test_update_progress_updates_title(monkeypatch):
    mock_root = MagicMock()
    mock_bar = MagicMock()
    # Mocking progress_bar['maximum'] = 200
    mock_bar.__getitem__.side_effect = lambda key: 200 if key == 'maximum' else None

    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "progress_bar", mock_bar)

    # 50 out of 200 is 25%
    gptscan.update_progress(50)

    mock_root.title.assert_called_with("[25%] GPT Virus Scanner")

def test_set_scanning_state_resets_title(monkeypatch):
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, "root", mock_root)

    # When scanning stops
    gptscan.set_scanning_state(False)

    mock_root.title.assert_called_with("GPT Virus Scanner")

def test_update_progress_handles_zero_division(monkeypatch):
    mock_root = MagicMock()
    mock_bar = MagicMock()
    # Mocking progress_bar['maximum'] = 0
    mock_bar.__getitem__.side_effect = lambda key: 0 if key == 'maximum' else None

    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "progress_bar", mock_bar)

    mock_root.title.reset_mock()
    gptscan.update_progress(50)

    # Should not raise exception and should not call title if division by zero would occur or max is 0
    # Actually my code does `if max_val > 0:`
    mock_root.title.assert_not_called()
