import pytest
from unittest.mock import patch, MagicMock
import gptscan
import subprocess

def test_get_git_config_snippets(mocker):
    # Mock subprocess.check_output to return specific git config output
    def mock_check_output(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "--is-inside-work-tree"]:
            return b"true"
        if cmd == ["git", "config", "--global", "--list"]:
            return "alias.co=checkout\nalias.dangerous=!rm -rf /\ncore.editor=vim\nuser.name=Test"
        if cmd == ["git", "config", "--local", "--list"]:
            return "core.sshcommand=ssh -i /path/to/key\nmerge.tool=kdiff3"
        return ""

    mocker.patch("subprocess.check_output", side_effect=mock_check_output)

    snippets = gptscan.get_git_config_snippets()

    # We expect:
    # 1. Global Alias: dangerous -> rm -rf /
    # 2. Global core.editor -> vim
    # 3. Local core.sshcommand -> ssh -i /path/to/key
    # Note: alias.co should be ignored because it doesn't start with !
    # Note: user.name and merge.tool should be ignored as they are not in dangerous_patterns

    assert len(snippets) == 3

    snippet_names = [s[0] for s in snippets]
    assert "Global Git Config [Alias: dangerous]" in snippet_names
    assert "Global Git Config [core.editor]" in snippet_names
    assert "Local Git Config [core.sshcommand]" in snippet_names

    snippet_contents = [s[1] for s in snippets]
    assert b"rm -rf /" in snippet_contents
    assert b"vim" in snippet_contents
    assert b"ssh -i /path/to/key" in snippet_contents

def test_scan_git_config_click(mocker):
    mocker.patch("gptscan.get_git_config_snippets", return_value=[("name", b"content")])
    mock_button_click = mocker.patch("gptscan.button_click")

    gptscan.scan_git_config_click()

    mock_button_click.assert_called_once_with(extra_snippets=[("name", b"content")])

def test_scan_git_config_click_empty(mocker):
    mocker.patch("gptscan.get_git_config_snippets", return_value=[])
    mock_messagebox = mocker.patch("tkinter.messagebox.showinfo")

    gptscan.scan_git_config_click()

    mock_messagebox.assert_called_once()
    assert "No potentially dangerous Git configuration settings" in mock_messagebox.call_args[0][1]

def test_system_audit_includes_git_config(mocker):
    mocker.patch("gptscan.get_shell_profile_paths", return_value=[])
    mocker.patch("gptscan.get_shell_history_paths", return_value=[])
    mocker.patch("gptscan.get_system_path_directories", return_value=[])
    mocker.patch("gptscan.get_ssh_config_paths", return_value=[])
    mocker.patch("gptscan.get_system_service_paths", return_value=[])
    mocker.patch("gptscan.get_git_hooks_paths", return_value=[])

    mocker.patch("gptscan.get_running_process_commands", return_value=[])
    mocker.patch("gptscan.get_environment_variable_snippets", return_value=[])
    mocker.patch("gptscan.get_scheduled_task_commands", return_value=[])
    mocker.patch("gptscan.get_startup_item_commands", return_value=[])
    mocker.patch("gptscan.get_system_service_commands", return_value=[])

    mocker.patch("gptscan.get_git_config_snippets", return_value=[("Git Config", b"Dangerous")])

    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")

    gptscan.scan_system_audit_click()

    mock_button_click.assert_called_once()
    _, kwargs = mock_button_click.call_args
    snippets = kwargs["extra_snippets"]
    assert ("Git Config", b"Dangerous") in snippets
