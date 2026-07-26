import sys
import subprocess
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_get_cli_clipboard_content_darwin_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    def mock_check_output(cmd, **kwargs):
        if cmd == ["pbpaste"]:
            return "darwin clipboard content"
        raise Exception("unexpected command")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = gptscan.get_cli_clipboard_content()
    assert result == "darwin clipboard content"

def test_get_cli_clipboard_content_darwin_failure_fallback_none(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    def mock_check_output(cmd, **kwargs):
        raise Exception("pbpaste failed")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # Prevent tkinter fallback from working
    with patch("tkinter.Tk", side_effect=Exception("no display")):
        result = gptscan.get_cli_clipboard_content()
        assert result is None

def test_get_cli_clipboard_content_win32_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    def mock_check_output(cmd, **kwargs):
        if "powershell.exe" in cmd:
            return "win32 clipboard content"
        raise Exception("unexpected command")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = gptscan.get_cli_clipboard_content()
    assert result == "win32 clipboard content"

def test_get_cli_clipboard_content_win32_failure_fallback_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    def mock_check_output(cmd, **kwargs):
        raise Exception("powershell failed")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    mock_tk = MagicMock()
    mock_tk_instance = MagicMock()
    mock_tk_instance.clipboard_get.return_value = "win32 tkinter content"
    mock_tk.return_value = mock_tk_instance

    with patch("tkinter.Tk", mock_tk):
        result = gptscan.get_cli_clipboard_content()
        assert result == "win32 tkinter content"

def test_get_cli_clipboard_content_linux_xclip_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    def mock_check_output(cmd, **kwargs):
        if "xclip" in cmd[0]:
            return "linux xclip content"
        raise Exception("unexpected command")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = gptscan.get_cli_clipboard_content()
    assert result == "linux xclip content"

def test_get_cli_clipboard_content_linux_xclip_failure_xsel_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    def mock_check_output(cmd, **kwargs):
        if "xclip" in cmd[0]:
            raise Exception("xclip missing")
        if "xsel" in cmd[0]:
            return "linux xsel content"
        raise Exception("unexpected command")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = gptscan.get_cli_clipboard_content()
    assert result == "linux xsel content"

def test_get_cli_clipboard_content_linux_all_tools_failed_fallback_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    def mock_check_output(cmd, **kwargs):
        raise Exception("tool missing")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    mock_tk = MagicMock()
    mock_tk_instance = MagicMock()
    mock_tk_instance.clipboard_get.return_value = "linux tkinter content"
    mock_tk.return_value = mock_tk_instance

    with patch("tkinter.Tk", mock_tk):
        result = gptscan.get_cli_clipboard_content()
        assert result == "linux tkinter content"
