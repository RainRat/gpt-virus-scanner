import pytest
from unittest.mock import patch, MagicMock
import subprocess
import json
import sys
import os
from gptscan import get_running_process_commands, scan_running_processes_click

def test_get_running_process_commands_linux():
    mock_output = "ARGS\n/usr/bin/python3 gptscan.py\n[kthreadd]\n/usr/lib/systemd/systemd --user\n"
    with patch("sys.platform", "linux"):
        with patch("subprocess.check_output", return_value=mock_output):
            processes = get_running_process_commands()

            # Should have python3 and systemd, but not kthreadd (kernel thread) or header
            assert len(processes) == 2
            assert processes[0][0] == "[Process] python3"
            assert processes[0][1] == b"/usr/bin/python3 gptscan.py"
            assert processes[1][0] == "[Process] systemd"
            assert processes[1][1] == b"/usr/lib/systemd/systemd --user"

def test_get_running_process_commands_windows():
    mock_json = json.dumps([
        {"Name": "python.exe", "CommandLine": "python gptscan.py"},
        {"Name": "svchost.exe", "CommandLine": None},
        {"Name": "cmd.exe", "CommandLine": "cmd /c dir"}
    ])
    with patch("sys.platform", "win32"):
        with patch("subprocess.check_output", return_value=mock_json):
            processes = get_running_process_commands()

            # Should have python.exe and cmd.exe, but not svchost.exe (no command line)
            assert len(processes) == 2
            assert processes[0][0] == "[Process] python.exe"
            assert processes[0][1] == b"python gptscan.py"
            assert processes[1][0] == "[Process] cmd.exe"
            assert processes[1][1] == b"cmd /c dir"

def test_get_running_process_commands_error():
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "ps")):
        processes = get_running_process_commands()
        assert processes == []

@patch("gptscan.get_running_process_commands")
@patch("gptscan.button_click")
def test_scan_running_processes_click_success(mock_button_click, mock_get_cmds):
    mock_get_cmds.return_value = [("[Process] test", b"test command")]

    scan_running_processes_click()

    mock_button_click.assert_called_once_with(extra_snippets=mock_get_cmds.return_value)

@patch("gptscan.get_running_process_commands")
@patch("gptscan.messagebox.showinfo")
def test_scan_running_processes_click_empty(mock_showinfo, mock_get_cmds):
    mock_get_cmds.return_value = []

    scan_running_processes_click()

    mock_showinfo.assert_called_once()
