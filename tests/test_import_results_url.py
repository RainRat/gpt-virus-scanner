import json
import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_import_from_url_success(monkeypatch):
    """Test importing scan results from a URL successfully."""
    mock_url = "https://example.com/results.json"
    data = [
        {
            "path": "test.py",
            "own_conf": "85%",
            "admin_desc": "Suspicious",
            "end-user_desc": "Don't run",
            "gpt_conf": "90%",
            "snippet": "print('hello')",
            "line": "10"
        }
    ]
    content = json.dumps(data).encode('utf-8')

    # Mock GUI components
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)
    monkeypatch.setattr(gptscan.tkinter.simpledialog, "askstring", lambda *args, **kwargs: mock_url)

    # Mock network call
    mock_fetch = MagicMock(return_value=content)
    monkeypatch.setattr(gptscan, "fetch_url_content", mock_fetch)

    # Mock internal helpers
    mock_finalize = MagicMock()
    monkeypatch.setattr(gptscan, "_finalize_import", mock_finalize)
    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_from_url()

    mock_fetch.assert_called_once_with(mock_url)
    mock_finalize.assert_called_once()
    args, _ = mock_finalize.call_args
    assert args[0][0]["path"] == "test.py"
    assert args[1] == mock_url

def test_import_from_url_cancelled(monkeypatch):
    """Test that cancelling the URL dialog does nothing."""
    monkeypatch.setattr(gptscan, "tree", MagicMock())
    monkeypatch.setattr(gptscan.tkinter.simpledialog, "askstring", lambda *args, **kwargs: None)

    mock_fetch = MagicMock()
    monkeypatch.setattr(gptscan, "fetch_url_content", mock_fetch)

    gptscan.import_from_url()
    mock_fetch.assert_not_called()

def test_import_from_url_error(monkeypatch):
    """Test error handling when fetching from URL fails."""
    mock_url = "https://example.com/results.json"
    monkeypatch.setattr(gptscan, "tree", MagicMock())
    monkeypatch.setattr(gptscan.tkinter.simpledialog, "askstring", lambda *args, **kwargs: mock_url)

    def mock_fetch_error(url):
        raise ValueError("Network error")

    monkeypatch.setattr(gptscan, "fetch_url_content", mock_fetch_error)

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)
    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    gptscan.import_from_url()

    mock_messagebox.showerror.assert_called()
    assert "Network error" in mock_messagebox.showerror.call_args[0][1]

def test_import_results_generator_url(monkeypatch):
    """Test the generator used by CLI for URL imports."""
    mock_url = "http://example.com/results.json"
    data = [{"path": "remote.py", "own_conf": "50%"}]
    content = json.dumps(data).encode('utf-8')

    mock_fetch = MagicMock(return_value=content)
    monkeypatch.setattr(gptscan, "fetch_url_content", mock_fetch)

    events = list(gptscan.import_results_generator(mock_url))

    # Check if we got the result event
    result_events = [e for e in events if e[0] == 'result']
    assert len(result_events) == 1
    assert result_events[0][1][0] == "remote.py"
    mock_fetch.assert_called_once_with(mock_url)

def test_import_results_generator_file(monkeypatch, tmp_path):
    """Verify that file imports still work with the updated generator."""
    local_file = tmp_path / "local.json"
    data = [{"path": "local.py", "own_conf": "10%"}]
    local_file.write_text(json.dumps(data))

    # Mock fetch to ensure it's NOT called for local files
    mock_fetch = MagicMock()
    monkeypatch.setattr(gptscan, "fetch_url_content", mock_fetch)

    events = list(gptscan.import_results_generator(str(local_file)))

    result_events = [e for e in events if e[0] == 'result']
    assert len(result_events) == 1
    assert result_events[0][1][0] == "local.py"
    mock_fetch.assert_not_called()
