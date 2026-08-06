import pytest
from unittest.mock import MagicMock
import gptscan

def test_toggle_dry_run_no_button_or_var(monkeypatch):
    monkeypatch.setattr(gptscan, "scan_button", None)
    monkeypatch.setattr(gptscan, "dry_var", None)

    gptscan.toggle_dry_run()

def test_toggle_dry_run_not_scanning_dry_run_true(monkeypatch):
    mock_button = MagicMock()
    mock_var = MagicMock()
    mock_var.get.return_value = True

    monkeypatch.setattr(gptscan, "scan_button", mock_button)
    monkeypatch.setattr(gptscan, "dry_var", mock_var)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    gptscan.toggle_dry_run()

    mock_button.config.assert_called_once_with(text="Dry Run")

def test_toggle_dry_run_not_scanning_dry_run_false(monkeypatch):
    mock_button = MagicMock()
    mock_var = MagicMock()
    mock_var.get.return_value = False

    monkeypatch.setattr(gptscan, "scan_button", mock_button)
    monkeypatch.setattr(gptscan, "dry_var", mock_var)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    gptscan.toggle_dry_run()

    mock_button.config.assert_called_once_with(text="Scan Now")

def test_toggle_dry_run_scanning_does_nothing(monkeypatch):
    mock_button = MagicMock()
    mock_var = MagicMock()
    mock_var.get.return_value = True
    mock_event = MagicMock()

    monkeypatch.setattr(gptscan, "scan_button", mock_button)
    monkeypatch.setattr(gptscan, "dry_var", mock_var)
    monkeypatch.setattr(gptscan, "current_cancel_event", mock_event)

    gptscan.toggle_dry_run()

    mock_button.config.assert_not_called()
