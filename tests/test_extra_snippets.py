import io
import subprocess
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import scan_files, Config, scan_clipboard_click

def test_scan_files_extra_snippets_basic(mock_tf_env, monkeypatch):
    """Test that extra_snippets are correctly processed by scan_files."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets, **kwargs: [])
    snippet_content = b"print('hello world')"
    extra_snippets = [("[Clipboard]", snippet_content)]

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        extra_snippets=extra_snippets
    ))

    # Filter for result events
    results = [data for event, data in events if event == 'result']

    assert len(results) == 1
    path, own_conf, admin, user, gpt, snippet, line = results[0]
    assert path == "[Clipboard]"
    assert own_conf == "50%"
    assert "print('hello world')" in snippet

def test_scan_files_boundary_threshold(mock_tf_env, monkeypatch):
    """Verify that a snippet exactly at the threshold is yielded (inclusive check)."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets, **kwargs: [])
    mock_tf_env.predict.return_value = [[0.5]] # Exactly 50%
    Config.THRESHOLD = 50

    snippet_content = b"at threshold"
    extra_snippets = [("boundary", snippet_content)]

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=False, # NOT showing all
        use_gpt=False,
        extra_snippets=extra_snippets
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    assert results[0][1] == "50%"

def test_scan_files_below_threshold_ignored(mock_tf_env, monkeypatch):
    """Verify that a snippet below the threshold is ignored when show_all=False."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets, **kwargs: [])
    mock_tf_env.predict.return_value = [[0.49]] # 49%
    Config.THRESHOLD = 50

    snippet_content = b"below threshold"
    extra_snippets = [("safe", snippet_content)]

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=False,
        use_gpt=False,
        extra_snippets=extra_snippets
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 0

def test_scan_files_extra_snippets_gpt_trigger(mock_tf_env, monkeypatch):
    """Test that extra_snippets can trigger GPT analysis."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets, **kwargs: [])
    mock_tf_env.predict.return_value = [[0.9]]
    Config.THRESHOLD = 50
    Config.GPT_ENABLED = True
    Config.taskdesc = "Analyze this"

    async def mock_gpt_handle(*args, **kwargs):
        return {
            "administrator": "Admin analysis",
            "end-user": "User explanation",
            "threat-level": 85
        }

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_gpt_handle)

    extra_snippets = [("[Stdin]", b"malicious code")]

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=False,
        use_gpt=True,
        extra_snippets=extra_snippets
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    path, own, admin, user, gpt, snippet, line = results[0]
    assert path == "[Stdin]"
    assert admin == "Admin analysis"
    assert gpt == "85%"

def test_cli_stdin_integration(monkeypatch):
    """Test that --stdin correctly passes content to run_cli."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    # Mock sys.stdin.buffer.read
    mock_stdin_buffer = MagicMock()
    mock_stdin_buffer.read.return_value = b"stdin content"

    # Mock sys.stdin and its buffer
    mock_stdin = MagicMock()
    mock_stdin.buffer = mock_stdin_buffer

    # Simulate: python gptscan.py --cli --stdin
    test_args = ["gptscan.py", "--cli", "--stdin"]
    with patch("sys.argv", test_args), patch("sys.stdin", mock_stdin):
        gptscan.main()

    mock_run_cli.assert_called_once()
    _, kwargs = mock_run_cli.call_args
    extra_snippets = kwargs.get("extra_snippets")
    assert extra_snippets == [("[Stdin]", b"stdin content")]

def test_scan_clipboard_click_logic(monkeypatch):
    """Test the GUI clipboard scan flow."""
    mock_root = MagicMock()
    mock_root.clipboard_get.return_value = "clipboard content"
    monkeypatch.setattr(gptscan, "root", mock_root)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    scan_clipboard_click()

    mock_button_click.assert_called_once_with(
        extra_snippets=[("[Clipboard]", b"clipboard content")]
    )

def test_scan_clipboard_click_empty(monkeypatch):
    """Test that empty clipboard does nothing."""
    mock_root = MagicMock()
    mock_root.clipboard_get.return_value = ""
    monkeypatch.setattr(gptscan, "root", mock_root)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    scan_clipboard_click()

    mock_button_click.assert_not_called()

def test_scan_clipboard_click_error(monkeypatch):
    """Test error handling in scan_clipboard_click."""
    mock_root = MagicMock()
    mock_root.clipboard_get.side_effect = Exception("Clipboard busy")
    monkeypatch.setattr(gptscan, "root", mock_root)

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    scan_clipboard_click()

    mock_messagebox.showwarning.assert_called_once()
    assert "Could not read from clipboard" in mock_messagebox.showwarning.call_args[0][1]

def test_get_cli_clipboard_content_darwin(monkeypatch):
    """Verify Darwin pbpaste is called and returns correct content."""
    monkeypatch.setattr("sys.platform", "darwin")
    mock_check = MagicMock(return_value="macOS clip")
    monkeypatch.setattr("subprocess.check_output", mock_check)

    content = gptscan.get_cli_clipboard_content()
    assert content == "macOS clip"
    mock_check.assert_called_with(['pbpaste'], text=True, stderr=subprocess.DEVNULL)

def test_get_cli_clipboard_content_win32(monkeypatch):
    """Verify Win32 powershell is called and returns correct content."""
    monkeypatch.setattr("sys.platform", "win32")
    mock_check = MagicMock(return_value="windows clip")
    monkeypatch.setattr("subprocess.check_output", mock_check)

    content = gptscan.get_cli_clipboard_content()
    assert content == "windows clip"
    mock_check.assert_called_with(['powershell.exe', '-NoProfile', '-Command', 'Get-Clipboard'], text=True, stderr=subprocess.DEVNULL)

def test_get_cli_clipboard_content_linux(monkeypatch):
    """Verify Linux xclip/xsel is called and returns correct content."""
    monkeypatch.setattr("sys.platform", "linux")
    mock_check = MagicMock(return_value="linux clip")
    monkeypatch.setattr("subprocess.check_output", mock_check)

    content = gptscan.get_cli_clipboard_content()
    assert content == "linux clip"
    mock_check.assert_any_call(['xclip', '-selection', 'clipboard', '-o'], text=True, stderr=subprocess.DEVNULL)

def test_get_cli_clipboard_content_fallback_tkinter(monkeypatch):
    """Verify Tkinter fallback is used when CLI tools fail."""
    monkeypatch.setattr("sys.platform", "linux")

    # Force CLI tools to fail
    mock_check = MagicMock(side_effect=Exception("no tools"))
    monkeypatch.setattr("subprocess.check_output", mock_check)

    # Mock tkinter.Tk
    mock_tk_inst = MagicMock()
    mock_tk_inst.clipboard_get.return_value = "tkinter fallback clip"
    mock_tk_class = MagicMock(return_value=mock_tk_inst)
    monkeypatch.setattr("tkinter.Tk", mock_tk_class)

    content = gptscan.get_cli_clipboard_content()
    assert content == "tkinter fallback clip"
    mock_tk_inst.clipboard_get.assert_called_once()
    mock_tk_inst.destroy.assert_called_once()

def test_get_cli_clipboard_content_failure(monkeypatch):
    """Verify None is returned when both CLI tools and Tkinter fail."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("subprocess.check_output", MagicMock(side_effect=Exception("no tools")))
    monkeypatch.setattr("tkinter.Tk", MagicMock(side_effect=Exception("no display")))

    content = gptscan.get_cli_clipboard_content()
    assert content is None

def test_cli_clipboard_integration(monkeypatch):
    """Test that CLI parameter --clipboard correctly fetches and passes clipboard to run_cli."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", MagicMock(return_value="my clip content"))

    test_args = ["gptscan.py", "--cli", "--clipboard"]
    with patch("sys.argv", test_args):
        gptscan.main()

    mock_run_cli.assert_called_once()
    _, kwargs = mock_run_cli.call_args
    extra_snippets = kwargs.get("extra_snippets")
    assert extra_snippets == [("[Clipboard]", b"my clip content")]
