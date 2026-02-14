import asyncio
import pytest
from unittest.mock import MagicMock, patch
import json
from tkinter import ttk
import gptscan
from gptscan import format_bytes, extract_data_from_gpt_response, async_handle_gpt_response, Config, AsyncRateLimiter, get_wrapped_values, enqueue_ui_update, ui_queue

def test_format_bytes_units():
    assert "100.0 B" in format_bytes(100)
    assert "1.0 KiB" in format_bytes(1024)
    assert "1.0 MiB" in format_bytes(1024**2)
    assert "1.0 GiB" in format_bytes(1024**3)
    assert "1.0 TiB" in format_bytes(1024**4)
    assert "1.0 PiB" in format_bytes(1024**5)
    assert "1.5 PiB" in format_bytes(1024**5 * 1.5)

def test_format_bytes_edge_cases():
    assert "0.0 B" in format_bytes(0)
    assert "-512.0 B" in format_bytes(-512)
    assert "-1.0 KiB" in format_bytes(-1024)

def test_extract_data_from_gpt_response_malformed_json():
    class MockChoice:
        def __init__(self, content):
            self.message = MagicMock(content=content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    response = MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 50')
    result = extract_data_from_gpt_response(response)

    assert isinstance(result, str)
    assert "Expecting" in result or "column" in result

@pytest.mark.asyncio
async def test_async_handle_gpt_response_no_client(monkeypatch):
    monkeypatch.setattr("gptscan.get_async_openai_client", lambda: None)
    monkeypatch.setattr(Config, "gpt_cache", {})

    limiter = AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(5)

    result = await async_handle_gpt_response(
        "some snippet",
        "some task",
        rate_limiter=limiter,
        semaphore=semaphore
    )

    assert result is None

def test_get_wrapped_values_with_hidden_column(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "c2", "c3", "c4", "c5", "c6", "orig_json") if key == "columns" else MagicMock()
    mock_tree.column.return_value = {"width": 100}

    orig_json_data = '{"key": "value"}'
    values = ["path/to/file", "50%", "admin", "user", "40%", "snippet", orig_json_data]

    monkeypatch.setattr("tkinter.font.Font", MagicMock())

    with patch("gptscan.adjust_newlines", side_effect=lambda v, w, measure=None: f"wrapped_{v}"):
        result = get_wrapped_values(mock_tree, values)

    assert len(result) == 7
    assert result[0] == "wrapped_path/to/file"
    assert result[6] == orig_json_data

def test_enqueue_ui_update():
    while not ui_queue.empty():
        ui_queue.get()

    def my_func(a, b=1):
        pass

    enqueue_ui_update(my_func, "arg1", b=2)

    assert not ui_queue.empty()
    func, args, kwargs = ui_queue.get()
    assert func == my_func
    assert args == ("arg1",)
    assert kwargs == {"b": 2}

def test_progress_utils(monkeypatch):
    mock_bar = MagicMock()
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, "progress_bar", mock_bar)
    monkeypatch.setattr(gptscan, "root", mock_root)

    gptscan.configure_progress(100)
    mock_bar.__setitem__.assert_any_call("maximum", 100)
    mock_bar.__setitem__.assert_any_call("value", 0)
    assert mock_root.update_idletasks.call_count >= 1

    mock_root.update_idletasks.reset_mock()
    gptscan.update_progress(50)
    mock_bar.__setitem__.assert_any_call("value", 50)
    assert mock_root.update_idletasks.call_count >= 1
