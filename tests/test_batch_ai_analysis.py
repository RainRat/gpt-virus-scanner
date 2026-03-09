import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import gptscan
import threading
import json
import asyncio
import time

@pytest.fixture
def mock_gui(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.Config, "taskdesc", "Test Task")

    # Mock UI updates to be synchronous for testing
    monkeypatch.setattr(gptscan, "enqueue_ui_update", lambda f, *args, **kwargs: f(*args, **kwargs))
    monkeypatch.setattr(gptscan, "update_status", MagicMock())
    monkeypatch.setattr(gptscan, "set_scanning_state", MagicMock())
    monkeypatch.setattr(gptscan, "update_tree_row", MagicMock())
    monkeypatch.setattr(gptscan, "finish_scan_state", MagicMock())

    return mock_tree

def test_analyze_selected_with_ai_initiation(mock_gui, monkeypatch):
    mock_tree = mock_gui
    mock_tree.selection.return_value = ["item1", "item2"]

    # Mock raw values
    # (path, own_conf, admin, user, gpt_conf, snippet, line)
    raw1 = ["file1.py", "80%", "", "", "", "print('evil')", "1"]
    raw2 = ["file2.py", "90%", "", "", "", "eval(x)", "10"]

    def mock_get_raw(iid):
        if iid == "item1": return raw1
        if iid == "item2": return raw2
        return None

    monkeypatch.setattr(gptscan, "_get_item_raw_values", mock_get_raw)

    mock_run_batch = MagicMock()
    monkeypatch.setattr(gptscan, "run_batch_ai_analysis", mock_run_batch)

    gptscan.analyze_selected_with_ai()

    assert gptscan.current_cancel_event is not None
    assert isinstance(gptscan.current_cancel_event, threading.Event)

    mock_run_batch.assert_called_once()
    args, _ = mock_run_batch.call_args
    requests, cancel_event = args
    assert len(requests) == 2
    assert requests[0]["path"] == "file1.py"
    assert requests[0]["item_id"] == "item1"
    assert requests[0]["cleaned_snippet"] == "print('evil')"
    assert requests[1]["path"] == "file2.py"
    assert requests[1]["item_id"] == "item2"
    assert cancel_event == gptscan.current_cancel_event

def test_analyze_selected_with_ai_disabled(mock_gui, monkeypatch):
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", False)
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_msgbox)

    gptscan.analyze_selected_with_ai()

    mock_msgbox.assert_called_once()
    assert "AI Disabled" in mock_msgbox.call_args[0][0]

def test_run_batch_ai_analysis_flow(mock_gui, monkeypatch):
    requests = [{"path": "file1.py", "item_id": "item1"}]
    cancel_event = threading.Event()

    # Mock the generator
    def mock_gen(*args, **kwargs):
        yield ('progress', (1, 1, "AI Analysis: file1.py"))
        # result_data = (path, own_conf, admin, user, gpt, snippet, line, item_id)
        yield ('result', ("file1.py", "80%", "Admin", "User", "90%", "Snippet", "1", "item1"))
        yield ('summary', (1, 0, 0.1))

    monkeypatch.setattr(gptscan, "batch_ai_analysis_events", mock_gen)

    gptscan.run_batch_ai_analysis(requests, cancel_event)

    gptscan.update_tree_row.assert_called_once()
    args, _ = gptscan.update_tree_row.call_args
    assert args[0] == "item1"
    # update_tree_row expects first 7 elements
    assert args[1] == ("file1.py", "80%", "Admin", "User", "90%", "Snippet", "1")

    gptscan.finish_scan_state.assert_called_once()

def test_batch_ai_analysis_events_loop(mock_gui, monkeypatch):
    # This test verifies batch_ai_analysis_events by mocking async_handle_gpt_response
    requests = [{
        "path": "file1.py",
        "percent": "80%",
        "snippet": "print('evil')",
        "cleaned_snippet": "print('evil')",
        "line": 1,
        "item_id": "item1"
    }]
    cancel_event = threading.Event()

    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.Config, "taskdesc", "Task")

    async def mock_handle(*args, **kwargs):
        return {
            "administrator": "Admin Note",
            "end-user": "User Note",
            "threat-level": 95
        }

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle)

    # Mock enqueue_ui_update for the notifier
    monkeypatch.setattr(gptscan, "enqueue_ui_update", MagicMock())

    gen = gptscan.batch_ai_analysis_events(requests, cancel_event)

    events = []
    # Collect events from generator
    for event in gen:
        events.append(event)
        if event[0] == 'summary':
            break

    # Verify expected events
    assert any(e[0] == 'progress' for e in events)
    result_event = next(e for e in events if e[0] == 'result')
    data = result_event[1]
    assert data[0] == "file1.py"
    assert data[2] == "Admin Note"
    assert data[4] == "95%"
    assert data[7] == "item1"

def test_batch_ai_analysis_events_ai_error(mock_gui, monkeypatch):
    requests = [{
        "path": "file1.py",
        "percent": "80%",
        "snippet": "code",
        "cleaned_snippet": "code",
        "item_id": "item1"
    }]
    cancel_event = threading.Event()

    async def mock_handle_error(*args, **kwargs):
        return None # Simulates AI failure

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle_error)
    monkeypatch.setattr(gptscan, "enqueue_ui_update", MagicMock())

    gen = gptscan.batch_ai_analysis_events(requests, cancel_event)

    events = list(gen)
    result_event = next(e for e in events if e[0] == 'result')
    data = result_event[1]
    assert data[2] == "AI Error"
    assert data[4] == "AI Error"

def test_batch_ai_analysis_events_cancellation(mock_gui, monkeypatch):
    requests = [{
        "path": "file1.py",
        "item_id": "item1",
        "snippet": "code",
        "cleaned_snippet": "code",
        "percent": "80%"
    }]
    cancel_event = threading.Event()

    async def mock_handle_cancel(*args, **kwargs):
        cancel_event.set() # Cancel during the AI call
        return None

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle_cancel)
    monkeypatch.setattr(gptscan, "enqueue_ui_update", MagicMock())

    gen = gptscan.batch_ai_analysis_events(requests, cancel_event)
    events = []
    for event in gen:
        events.append(event)
        if event[0] == 'summary':
            break

    # Verify that results are marked as 'Cancelled'
    result_events = [e for e in events if e[0] == 'result']
    assert len(result_events) == 1
    assert result_events[0][1][2] == "Cancelled"
