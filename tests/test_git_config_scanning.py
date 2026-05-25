import pytest
from unittest.mock import patch, MagicMock
import gptscan
import subprocess
import tkinter.messagebox

def test_get_git_config_snippets(monkeypatch):
    # Mock subprocess.check_output to return specific git config output
    def mock_check_output(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"] and "--is-inside-work-tree" in cmd:
            return b"true"
        if cmd[:3] == ["git", "config", "--global"] and "--list" in cmd and "-z" in cmd:
            # key\nvalue\0...
            return "alias.co\ncheckout\0alias.dangerous\n!rm -rf /\0core.editor\nvim\0user.name\nTest\0"
        if cmd[:3] == ["git", "config", "--local"] and "--list" in cmd and "-z" in cmd:
            return "core.sshcommand\nssh -i /path/to/key\0merge.tool\nkdiff3\0"
        return ""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    snippets = gptscan.get_git_config_snippets()

    snippet_names = [s[0] for s in snippets]
    assert "Global Git Config [Alias: dangerous]" in snippet_names
    assert "Global Git Config [core.editor]" in snippet_names
    assert "Local Git Config [core.sshcommand]" in snippet_names

    snippet_contents = [s[1] for s in snippets]
    assert b"rm -rf /" in snippet_contents
    assert b"vim" in snippet_contents
    assert b"ssh -i /path/to/key" in snippet_contents

def test_get_git_config_snippets_multiline(monkeypatch):
    def mock_check_output(cmd, **kwargs):
        if "-z" in cmd:
            if "--global" in cmd:
                return "alias.multi\n!echo line1\nline2\0core.editor\nvim\0"
            if "--local" in cmd:
                return ""
        if "rev-parse" in cmd:
            return b"true"
        return b""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    snippets = gptscan.get_git_config_snippets()
    multi_snippet = next((s[1].decode() for s in snippets if "multi" in s[0]), None)
    assert multi_snippet == "echo line1\nline2"

def test_scan_git_config_click(monkeypatch):
    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [("name", b"content")])
    mock_button_click = MagicMock()
    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_git_config_click()

    mock_button_click.assert_called_once_with(extra_snippets=[("name", b"content")])

def test_scan_git_config_click_empty(monkeypatch):
    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [])
    mock_showinfo = MagicMock()
    monkeypatch.setattr(tkinter.messagebox, "showinfo", mock_showinfo)

    gptscan.scan_git_config_click()

    mock_showinfo.assert_called_once()
    assert "No potentially dangerous Git configuration settings" in mock_showinfo.call_args[0][1]

def test_system_audit_includes_git_config(monkeypatch):
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_shell_history_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_path_directories", lambda: [])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: [])

    monkeypatch.setattr("gptscan.get_running_process_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_environment_variable_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_scheduled_task_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_startup_item_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_commands", lambda: [])

    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [("Git Config", b"Dangerous")])

    monkeypatch.setattr("gptscan._set_scan_target", MagicMock())
    mock_button_click = MagicMock()
    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_system_audit_click()

    mock_button_click.assert_called_once()
    _, kwargs = mock_button_click.call_args
    snippets = kwargs["extra_snippets"]
    assert ("Git Config", b"Dangerous") in snippets
