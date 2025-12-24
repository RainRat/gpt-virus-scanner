import sys
from unittest.mock import MagicMock
import pytest
import gptscan
import tkinter.filedialog

def test_browse_button_click_cancels_does_not_clear_textbox(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

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
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    # Mock askdirectory to return a path
    monkeypatch.setattr(tkinter.filedialog, 'askdirectory', lambda: '/path/to/folder')

    # Call function
    gptscan.browse_button_click()

    # Assert
    # We expect delete(0, END) and insert(0, path)
    mock_textbox.delete.assert_called_with(0, gptscan.tk.END)
    mock_textbox.insert.assert_called_with(0, '/path/to/folder')

def test_set_scanning_state_updates_buttons(monkeypatch):
    # Setup mocks
    mock_scan_button = MagicMock()
    mock_cancel_button = MagicMock()
    monkeypatch.setattr(gptscan, 'scan_button', mock_scan_button, raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', mock_cancel_button, raising=False)

    # Test scanning=True
    gptscan.set_scanning_state(True)
    mock_scan_button.config.assert_called_with(state="disabled")
    mock_cancel_button.config.assert_called_with(state="normal")

    # Test scanning=False
    gptscan.set_scanning_state(False)
    mock_scan_button.config.assert_called_with(state="normal")
    mock_cancel_button.config.assert_called_with(state="disabled")

def test_finish_scan_state_resets_state(monkeypatch):
    # Setup
    mock_event = MagicMock()
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)
    mock_scan_button = MagicMock()
    mock_cancel_button = MagicMock()
    monkeypatch.setattr(gptscan, 'scan_button', mock_scan_button, raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', mock_cancel_button, raising=False)

    # Mock status_label
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    # Action
    gptscan.finish_scan_state()

    # Assert
    mock_status_label.config.assert_called_with(text="Ready")
    assert gptscan.current_cancel_event is None
    mock_scan_button.config.assert_called_with(state="normal")
    mock_cancel_button.config.assert_called_with(state="disabled")

def test_cancel_scan_triggers_event(monkeypatch):
    # Setup
    mock_event = MagicMock()
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)

    # Action
    gptscan.cancel_scan()

    # Assert
    mock_event.set.assert_called_once()

def test_cancel_scan_does_nothing_if_no_event(monkeypatch):
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    # Should not raise
    gptscan.cancel_scan()

def test_button_click_validation_failure(monkeypatch):
    # Mock textbox to return empty
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    gptscan.button_click()

    mock_msgbox.showerror.assert_called_with("Scan Error", "Please select a directory to scan.")

def test_button_click_missing_model(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/some/path"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Mock os.path.exists to fail for scripts.h5
    monkeypatch.setattr(gptscan.os.path, 'exists', lambda p: False)

    gptscan.button_click()

    mock_msgbox.showerror.assert_called_with("Scan Error", "Model file scripts.h5 not found.")

def test_button_click_starts_scan(monkeypatch):
    # Setup
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/some/path"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    mock_scan_button = MagicMock()
    mock_cancel_button = MagicMock()
    monkeypatch.setattr(gptscan, 'scan_button', mock_scan_button, raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', mock_cancel_button, raising=False)

    # Mock vars
    mock_deep = MagicMock()
    mock_deep.get.return_value = True
    monkeypatch.setattr(gptscan, 'deep_var', mock_deep, raising=False)

    mock_all = MagicMock()
    mock_all.get.return_value = False
    monkeypatch.setattr(gptscan, 'all_var', mock_all, raising=False)

    mock_gpt = MagicMock()
    mock_gpt.get.return_value = True
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt, raising=False)

    # Mock os.path.exists
    monkeypatch.setattr(gptscan.os.path, 'exists', lambda p: True)

    # Mock Thread
    mock_thread_cls = MagicMock()
    mock_thread_instance = MagicMock()
    mock_thread_cls.return_value = mock_thread_instance
    monkeypatch.setattr(gptscan.threading, 'Thread', mock_thread_cls)

    # Mock status_label
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    # Action
    gptscan.button_click()

    # Assert
    mock_status_label.config.assert_called_with(text="Starting scan...")
    assert gptscan.current_cancel_event is not None
    mock_scan_button.config.assert_called_with(state="disabled")

    # Check thread creation
    mock_thread_cls.assert_called_once()
    _, kwargs = mock_thread_cls.call_args
    assert kwargs['daemon'] is True
    assert kwargs['target'] == gptscan.run_scan

    args = kwargs['args']
    assert args[0] == "/some/path"
    assert args[1] is True # deep
    assert args[2] is False # all
    assert args[3] is True # gpt
    assert args[4] == gptscan.current_cancel_event

    mock_thread_instance.start.assert_called_once()

def test_button_click_ignored_if_already_running(monkeypatch):
    mock_event = MagicMock()
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)

    mock_thread_cls = MagicMock()
    monkeypatch.setattr(gptscan.threading, 'Thread', mock_thread_cls)

    gptscan.button_click()

    mock_thread_cls.assert_not_called()
