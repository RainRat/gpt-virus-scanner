import pytest
from unittest.mock import MagicMock, patch
import gptscan
import asyncio
import threading
import tkinter as tk

def test_request_single_gpt_analysis_success(monkeypatch):
    """Test request_single_gpt_analysis with a successful AI response."""
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.Config, "taskdesc", "test task")

    expected_result = {"administrator": "admin note", "end-user": "user note", "threat-level": 10}

    async def mock_async_handle(*args, **kwargs):
        return expected_result

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_async_handle)

    result = gptscan.request_single_gpt_analysis("test snippet")
    assert result == expected_result

def test_request_single_gpt_analysis_disabled(monkeypatch):
    """Test request_single_gpt_analysis when GPT is disabled."""
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", False)

    result = gptscan.request_single_gpt_analysis("test snippet")
    assert result is None

def test_request_single_gpt_analysis_error(monkeypatch):
    """Test request_single_gpt_analysis when an exception occurs."""
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.Config, "taskdesc", "test task")

    async def mock_async_handle(*args, **kwargs):
        raise Exception("API Error")

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_async_handle)

    # It should catch the exception and return None
    result = gptscan.request_single_gpt_analysis("test snippet")
    assert result is None

def test_on_analyze_now_ui_flow(monkeypatch):
    """Test the 'Analyze with AI' button flow in the details window."""
    # 1. Setup Config and basic UI mocks
    monkeypatch.setattr(gptscan, "current_cancel_event", None)
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan, "root", MagicMock())

    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Mock _get_item_raw_values
    raw_vals = ["test.py", "50%", "", "", "", "print('evil')"]
    monkeypatch.setattr(gptscan, "_get_item_raw_values", lambda iid: raw_vals)

    # 2. Mock Buttons to capture the command
    captured_commands = {}
    mock_analyze_btn = MagicMock()

    # We need to capture the command passed to the button
    # on_analyze_now is passed as command=on_analyze_now
    def mock_button_init_with_cmd(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text == "Analyze with AI":
            captured_commands["Analyze with AI"] = kwargs.get('command')
            return mock_analyze_btn
        return btn

    monkeypatch.setattr(gptscan.ttk, "Button", mock_button_init_with_cmd)

    # 3. Mock threading to run synchronously
    def mock_thread_init(self, target=None, args=(), kwargs=None, **other):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
    def mock_thread_start(self):
        self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(threading.Thread, "__init__", mock_thread_init)
    monkeypatch.setattr(threading.Thread, "start", mock_thread_start)

    # 4. Mock UI update helpers
    monkeypatch.setattr(gptscan, "enqueue_ui_update", lambda f, *args, **kwargs: f(*args, **kwargs))
    mock_update_tree_row = MagicMock()
    monkeypatch.setattr(gptscan, "update_tree_row", mock_update_tree_row)

    # 5. Mock request_single_gpt_analysis
    ai_result = {"administrator": "admin note", "end-user": "user note", "threat-level": 90}
    monkeypatch.setattr(gptscan, "request_single_gpt_analysis", lambda s: ai_result)

    # 6. Mock ScrolledText and other Toplevel components
    monkeypatch.setattr(gptscan.tk, "Toplevel", MagicMock())
    monkeypatch.setattr(gptscan.scrolledtext, "ScrolledText", MagicMock())

    # 7. Execute view_details to instantiate the inner function and button
    gptscan.view_details(item_id="item1")

    assert "Analyze with AI" in captured_commands
    on_analyze_now = captured_commands["Analyze with AI"]

    # 8. Trigger the analysis
    on_analyze_now()

    # 9. Verify results
    # update_tree_row should be called with updated values
    mock_update_tree_row.assert_called_once()
    args, _ = mock_update_tree_row.call_args
    target_id, updated_vals = args
    assert target_id == "item1"
    assert updated_vals[2] == "admin note"
    assert updated_vals[3] == "user note"
    assert updated_vals[4] == "90%" # 90/100 formatted as %
