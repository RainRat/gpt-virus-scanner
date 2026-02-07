import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import gptscan

@pytest.fixture
def mock_gpt_env(monkeypatch, tmp_path):
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {})
    monkeypatch.setattr(gptscan, "_async_openai_client", None)

    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.Config, "taskdesc", "Test Task")
    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})

    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.9]]
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    mock_tf = MagicMock()
    mock_tf.constant = lambda x: x
    mock_tf.expand_dims = lambda x, axis: x
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    file1 = tmp_path / "test1.py"
    file1.write_text("print('hello')")
    file2 = tmp_path / "test2.py"
    file2.write_text("print('world')")
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [file1, file2])

    return file1, file2

def test_scan_files_gpt_processing_flow(mock_gpt_env, monkeypatch):
    file1, file2 = mock_gpt_env

    async def mock_handle(snippet, task, rate_limiter, semaphore, wait_callback=None):
        if "hello" in snippet:
            return {
                "administrator": "Admin Hello",
                "end-user": "User Hello",
                "threat-level": 90
            }
        return None

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle)

    results = list(gptscan.scan_files(
        scan_targets="dummy",
        deep_scan=False,
        show_all=False,
        use_gpt=True,
        rate_limit=60
    ))

    gpt_results = [r for t, r in results if t == 'result']
    assert len(gpt_results) == 2

    res1 = next(r for r in gpt_results if "test1.py" in r[0])
    path1, own1, admin1, user1, gpt1, snip1 = res1
    assert admin1 == "Admin Hello"
    assert gpt1 == "90%"

    res2 = next(r for r in gpt_results if "test2.py" in r[0])
    path2, own2, admin2, user2, gpt2, snip2 = res2
    assert admin2 == "JSON Parse Error"
    assert gpt2 == "JSON Parse Error"

def test_scan_files_gpt_rate_limit_notifier(mock_gpt_env, monkeypatch):
    async def mock_handle(snippet, task, rate_limiter, semaphore, wait_callback=None):
        if wait_callback:
            wait_callback(0.1)
        return {
            "administrator": "Admin",
            "end-user": "User",
            "threat-level": 50
        }

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    results = list(gptscan.scan_files(
        scan_targets="dummy",
        deep_scan=False,
        show_all=False,
        use_gpt=True,
        rate_limit=1
    ))

    status_messages = [r[2] for t, r in results if t == 'progress' and r[2] is not None]
    assert "Waiting for API rate limit..." in status_messages

def test_scan_files_gpt_cancellation_during_result_processing(mock_gpt_env, monkeypatch):
    file1, file2 = mock_gpt_env
    cancel_event = MagicMock()

    async def mock_handle(snippet, task, rate_limiter, semaphore, wait_callback=None):
        cancel_event.set_to_true = True
        return {"administrator": "A", "end-user": "U", "threat-level": 50}

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_handle)

    cancel_event.set_to_true = False
    def mock_is_set():
        return getattr(cancel_event, "set_to_true", False)

    cancel_event.is_set.side_effect = mock_is_set

    results = list(gptscan.scan_files(
        scan_targets="dummy",
        deep_scan=False,
        show_all=False,
        use_gpt=True,
        cancel_event=cancel_event
    ))

    gpt_results = [r for t, r in results if t == 'result']
    assert len(gpt_results) < 2

def test_scan_files_gpt_cancellation_after_local_scan(mock_gpt_env, monkeypatch):
    file1, file2 = mock_gpt_env
    cancel_event = MagicMock()

    scan_counts = 0
    def mock_is_set():
        nonlocal scan_counts
        # After both files are scanned (usually 2 checks per file), trigger cancellation
        # We can detect this by checking how many files were processed.
        # But a simpler way is to count calls and know that we want to cancel before GPT.
        # However, the reviewer wanted a more robust condition.
        return scan_counts >= 4

    def count_is_set():
        nonlocal scan_counts
        res = mock_is_set()
        scan_counts += 1
        return res

    cancel_event.is_set.side_effect = count_is_set

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", AsyncMock())

    results = list(gptscan.scan_files(
        scan_targets="dummy",
        deep_scan=False,
        show_all=False,
        use_gpt=True,
        cancel_event=cancel_event
    ))

    gpt_results = [r for t, r in results if t == 'result']
    assert len(gpt_results) == 0
