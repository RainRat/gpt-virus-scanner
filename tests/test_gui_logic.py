import sys
from unittest.mock import MagicMock
import pytest
import gptscan
import tkinter.filedialog

def test_browse_button_click_cancels_does_not_clear_textbox(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    # Inject textbox into gptscan module
    gptscan.textbox = mock_textbox

    # Mock askdirectory to return empty string (cancellation)
    monkeypatch.setattr(tkinter.filedialog, 'askdirectory', lambda: '')

    # Call function
    gptscan.browse_button_click()

    # Assert
    mock_textbox.delete.assert_not_called()
    mock_textbox.insert.assert_not_called()

def test_browse_button_click_selects_folder_updates_textbox(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    gptscan.textbox = mock_textbox

    # Mock askdirectory to return a path
    monkeypatch.setattr(tkinter.filedialog, 'askdirectory', lambda: '/path/to/folder')

    # Call function
    gptscan.browse_button_click()

    # Assert
    # We expect delete(0, END) and insert(0, path)
    mock_textbox.delete.assert_called_with(0, gptscan.tk.END)
    mock_textbox.insert.assert_called_with(0, '/path/to/folder')
