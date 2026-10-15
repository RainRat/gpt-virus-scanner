import sys
import threading
from unittest.mock import MagicMock
import pytest
import gptscan


def test_consume_scan_events_progress_events(monkeypatch, capsys):
    updates = []

    def mock_enqueue(func, *args, **kwargs):
        updates.append((func, args, kwargs))

    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    result_handler = MagicMock(return_value=True)

    events = [
        ('progress', (1, 10, "Scanning file 1")),
        ('progress', (2, 10, "")),
    ]

    gptscan._consume_scan_events(
        event_gen=events,
        cancel_event=cancel_event,
        status_prefix="Scanning",
        result_handler=result_handler,
        print_status=True,
    )

    captured = capsys.readouterr()
    assert "Scanning file 1 (1/10)" in captured.err
    assert "Scanning: 2/10" in captured.err

    funcs = [u[0] for u in updates]
    assert gptscan.configure_progress in funcs
    assert gptscan.update_progress in funcs
    assert gptscan.update_status in funcs
    assert gptscan.finish_scan_state in funcs


def test_consume_scan_events_results_and_threat_counts(monkeypatch):
    updates = []

    def mock_enqueue(func, *args, **kwargs):
        updates.append((func, args, kwargs))

    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()

    # Result handler returns True
    result_handler = MagicMock(return_value=True)

    # 1st item: High risk (effective conf 90%)
    # 2nd item: Medium risk (effective conf 60%)
    # 3rd item: Low risk (effective conf 20%)
    events = [
        ('result', ("file1.py", "90%", "desc", "user_desc", None, "snip")),
        ('result', ("file2.py", "60%", "desc", "user_desc", None, "snip")),
        ('result', ("file3.py", "20%", "desc", "user_desc", None, "snip")),
        ('progress', (3, 3, "Done")),
    ]

    gptscan._consume_scan_events(
        event_gen=events,
        cancel_event=cancel_event,
        status_prefix="Scanning",
        result_handler=result_handler,
        fail_threshold=30,
    )

    # Verify status text with threat suffix
    status_updates = [u for u in updates if u[0] == gptscan.update_status]
    last_status = status_updates[-1][1][0]
    assert "2 suspicious: 1 high, 1 medium" in last_status

    # Verify finish_scan_state arguments: (scanned, threats_found, bytes, elapsed, high, medium)
    finish_calls = [u for u in updates if u[0] == gptscan.finish_scan_state]
    assert len(finish_calls) == 1
    args = finish_calls[0][1]
    assert args == (3, 2, None, None, 1, 1)


def test_consume_scan_events_cancellation(monkeypatch):
    updates = []

    def mock_enqueue(func, *args, **kwargs):
        updates.append((func, args, kwargs))

    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    cancel_event.set()

    result_handler = MagicMock(return_value=True)

    events = [
        ('progress', (5, 10, "In Progress")),
        ('result', ("file1.py", "90%", "desc", "user_desc", None, "snip")),
    ]

    gptscan._consume_scan_events(
        event_gen=events,
        cancel_event=cancel_event,
        status_prefix="Scanning",
        result_handler=result_handler,
    )

    # result_handler should not be called because cancel_event is set
    result_handler.assert_not_called()

    status_updates = [u for u in updates if u[0] == gptscan.update_status]
    cancelled_status = [u[1][0] for u in status_updates if "cancelled" in u[1][0]]
    assert len(cancelled_status) == 1
    assert "Scan cancelled after 5 files." in cancelled_status[0]


def test_consume_scan_events_summary_event(monkeypatch):
    updates = []

    def mock_enqueue(func, *args, **kwargs):
        updates.append((func, args, kwargs))

    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    cancel_event = threading.Event()
    result_handler = MagicMock(return_value=True)

    events = [
        ('summary', (50, 10240, 2.5)),
    ]

    gptscan._consume_scan_events(
        event_gen=events,
        cancel_event=cancel_event,
        status_prefix="Scanning",
        result_handler=result_handler,
    )

    finish_calls = [u for u in updates if u[0] == gptscan.finish_scan_state]
    assert len(finish_calls) == 1
    args = finish_calls[0][1]
    # (total_files, threats_found, total_bytes, elapsed_time, high_risk_found, medium_risk_found)
    assert args == (50, 0, 10240, 2.5, 0, 0)
