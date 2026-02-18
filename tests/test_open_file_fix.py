import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk

def test_open_file_with_explicit_path(monkeypatch):
    """Test that open_file uses the provided path when passed as a string."""
    mock_selected = MagicMock(return_value=["selected.py"])
    monkeypatch.setattr(gptscan, '_get_selected_row_values', mock_selected)

    mock_exists = MagicMock(return_value=True)
    monkeypatch.setattr(gptscan.os.path, 'exists', mock_exists)

    # Mock subprocess.run based on platform
    mock_run = MagicMock()
    monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)

    if gptscan.sys.platform == "win32":
        mock_startfile = MagicMock()
        monkeypatch.setattr(gptscan.os, 'startfile', mock_startfile)
        gptscan.open_file("explicit.py")
        mock_startfile.assert_called_with("explicit.py")
    elif gptscan.sys.platform == "darwin":
        gptscan.open_file("explicit.py")
        mock_run.assert_called_with(["open", "explicit.py"])
    else:
        gptscan.open_file("explicit.py")
        mock_run.assert_called_with(["xdg-open", "explicit.py"])

    # Verify _get_selected_row_values was NOT called when path was provided
    assert mock_selected.call_count == 0

def test_open_file_with_none_uses_selection(monkeypatch):
    """Test that open_file falls back to selection when no path is provided."""
    mock_selected = MagicMock(return_value=["selected.py"])
    monkeypatch.setattr(gptscan, '_get_selected_row_values', mock_selected)

    mock_exists = MagicMock(return_value=True)
    monkeypatch.setattr(gptscan.os.path, 'exists', mock_exists)

    # Mock subprocess.run based on platform
    mock_run = MagicMock()
    monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)

    if gptscan.sys.platform == "win32":
        mock_startfile = MagicMock()
        monkeypatch.setattr(gptscan.os, 'startfile', mock_startfile)
        gptscan.open_file()
        mock_startfile.assert_called_with("selected.py")
    elif gptscan.sys.platform == "darwin":
        gptscan.open_file()
        mock_run.assert_called_with(["open", "selected.py"])
    else:
        gptscan.open_file()
        mock_run.assert_called_with(["xdg-open", "selected.py"])

    # Verify _get_selected_row_values WAS called
    assert mock_selected.call_count == 1

def test_open_file_with_event_uses_selection(monkeypatch):
    """Test that open_file falls back to selection when an event is provided."""
    mock_selected = MagicMock(return_value=["selected.py"])
    monkeypatch.setattr(gptscan, '_get_selected_row_values', mock_selected)

    mock_exists = MagicMock(return_value=True)
    monkeypatch.setattr(gptscan.os.path, 'exists', mock_exists)

    # Mock subprocess.run based on platform
    mock_run = MagicMock()
    monkeypatch.setattr(gptscan.subprocess, 'run', mock_run)

    mock_event = MagicMock()

    if gptscan.sys.platform == "win32":
        mock_startfile = MagicMock()
        monkeypatch.setattr(gptscan.os, 'startfile', mock_startfile)
        gptscan.open_file(mock_event)
        mock_startfile.assert_called_with("selected.py")
    elif gptscan.sys.platform == "darwin":
        gptscan.open_file(mock_event)
        mock_run.assert_called_with(["open", "selected.py"])
    else:
        gptscan.open_file(mock_event)
        mock_run.assert_called_with(["xdg-open", "selected.py"])

    # Verify _get_selected_row_values WAS called
    assert mock_selected.call_count == 1
