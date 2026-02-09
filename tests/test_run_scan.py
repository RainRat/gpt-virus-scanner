import threading
from unittest.mock import MagicMock
import pytest
import gptscan

def test_run_scan_basic_flow(monkeypatch):
    mock_scan_files = MagicMock()
    mock_scan_files.return_value = [
        ('progress', (0, 1, "Scanning...")),
        ('result', ("test.py", "80%", "Admin", "User", "90%", "code")),
        ('progress', (1, 1, None)),
        ('summary', (1, 1024, 1.5))
    ]
    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    mock_enqueue = MagicMock()
    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    gptscan.run_scan("dummy", False, True, True, cancel_event)

    calls = mock_enqueue.call_args_list

    assert calls[0][0][0] == gptscan.configure_progress
    assert calls[0][0][1] == 1

    assert calls[1][0][0] == gptscan.update_progress
    assert calls[1][0][1] == 0

    assert calls[2][0][0] == gptscan.update_status
    assert calls[2][0][1] == "Scanning... (0/1)"

    assert calls[3][0][0] == gptscan.insert_tree_row
    assert calls[3][0][1][0] == "test.py"

    assert calls[4][0][0] == gptscan.update_progress
    assert calls[4][0][1] == 1

    assert calls[5][0][0] == gptscan.update_status
    assert calls[5][0][1] == "Scanning: 1/1"

    assert calls[6][0][0] == gptscan.finish_scan_state
    assert calls[6][0][1] == 1
    assert calls[6][0][2] == 1
    assert calls[6][0][3] == 1024
    assert calls[6][0][4] == 1.5

def test_run_scan_cancellation(monkeypatch):
    cancel_event = threading.Event()

    def mock_scan_gen(*args, **kwargs):
        yield ('progress', (0, 10, "Scanning..."))
        cancel_event.set()
        yield ('result', ("test.py", "80%", "Admin", "User", "90%", "code"))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_gen)

    mock_enqueue = MagicMock()
    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    gptscan.run_scan("dummy", False, True, True, cancel_event)

    enqueued_funcs = [call[0][0] for call in mock_enqueue.call_args_list]
    assert gptscan.insert_tree_row not in enqueued_funcs
    assert gptscan.finish_scan_state in enqueued_funcs

def test_run_scan_gpt_json_error(monkeypatch):
    mock_scan_files = MagicMock()
    mock_scan_files.return_value = [
        ('result', ("test.py", "80%", "JSON Parse Error", "JSON Parse Error", "JSON Parse Error", "code")),
    ]
    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    mock_enqueue = MagicMock()
    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    gptscan.run_scan("dummy", False, True, True, cancel_event)

    calls = mock_enqueue.call_args_list
    finish_call = next(c for c in calls if c[0][0] == gptscan.finish_scan_state)
    assert finish_call[0][2] == 1

def test_run_scan_zero_files(monkeypatch):
    mock_scan_files = MagicMock()
    mock_scan_files.return_value = [
        ('progress', (0, 0, "Scanning...")),
        ('summary', (0, 0, 0.1))
    ]
    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    mock_enqueue = MagicMock()
    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    gptscan.run_scan("dummy", False, True, True, cancel_event)

    calls = mock_enqueue.call_args_list
    enqueued_funcs = [call[0][0] for call in calls]

    assert gptscan.configure_progress not in enqueued_funcs
    assert gptscan.update_progress in enqueued_funcs

    finish_call = next(c for c in calls if c[0][0] == gptscan.finish_scan_state)
    assert finish_call[0][1] == 0
    assert finish_call[0][2] == 0
