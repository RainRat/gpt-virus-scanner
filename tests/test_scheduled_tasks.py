import pytest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys
from gptscan import get_scheduled_task_commands, scan_scheduled_tasks_click

def test_get_scheduled_task_commands_linux():
    mock_crontab = "# Some comment\n* * * * * /usr/bin/python3 /path/to/script.py\n@daily /usr/bin/cleanup.sh\n"
    with patch("sys.platform", "linux"):
        with patch("subprocess.check_output", return_value=mock_crontab):
            tasks = get_scheduled_task_commands()

            assert len(tasks) == 2
            assert tasks[0][0] == "[Cron] User"
            assert tasks[0][1] == b"/usr/bin/python3 /path/to/script.py"
            assert tasks[1][0] == "[Cron] User"
            assert tasks[1][1] == b"/usr/bin/cleanup.sh"

def test_get_scheduled_task_commands_windows():
    # schtasks /query /fo CSV /v output
    mock_csv = '"HostName","TaskName","Next Run Time","Status","Last Run Time","Last Result","Creator","Schedule","Task To Run","Path","Run As User"\n' \
               '"HOST","\\MyTask","N/A","Ready","N/A","0","User","One Time Only","C:\\Windows\\System32\\calc.exe","\\","User"\n' \
               '"HOST","\\AnotherTask","N/A","Disabled","N/A","0","User","N/A","N/A","\\","User"\n'

    with patch("sys.platform", "win32"):
        with patch("subprocess.check_output", return_value=mock_csv):
            tasks = get_scheduled_task_commands()

            assert len(tasks) == 1
            assert tasks[0][0] == "[Task] \\MyTask"
            assert tasks[0][1] == b"C:\\Windows\\System32\\calc.exe"

def test_get_scheduled_task_commands_error():
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "crontab")):
        tasks = get_scheduled_task_commands()
        assert tasks == []

@patch("gptscan.get_scheduled_task_commands")
@patch("gptscan.button_click")
def test_scan_scheduled_tasks_click_success(mock_button_click, mock_get_tasks):
    mock_get_tasks.return_value = [("[Task] test", b"test command")]

    scan_scheduled_tasks_click()

    mock_button_click.assert_called_once_with(extra_snippets=mock_get_tasks.return_value)

@patch("gptscan.get_scheduled_task_commands")
@patch("gptscan.messagebox.showinfo")
def test_scan_scheduled_tasks_click_empty(mock_showinfo, mock_get_tasks):
    mock_get_tasks.return_value = []

    scan_scheduled_tasks_click()

    mock_showinfo.assert_called_once()
