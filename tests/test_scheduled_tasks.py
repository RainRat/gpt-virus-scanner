import pytest
from unittest.mock import patch, MagicMock, mock_open
import subprocess
import os
from pathlib import Path
from gptscan import get_scheduled_task_commands, scan_scheduled_tasks_click

def test_get_scheduled_task_commands_linux_all():
    mock_crontab_l = "# Some comment\n* * * * * /usr/bin/python3 /path/to/script.py\n@daily /usr/bin/cleanup.sh\n"
    mock_etc_crontab = "17 *	* * *	root    cd / && run-parts --report /etc/cron.hourly\n"
    mock_cron_d_file = "30 4	* * *	user    /usr/local/bin/daily_backup.sh\n@reboot root /usr/bin/reboot_task\n"

    with patch("sys.platform", "linux"):
        with patch("subprocess.check_output", return_value=mock_crontab_l):
            with patch("os.path.exists") as mock_exists:
                # We want to mock exists for /etc/crontab and /etc/cron.d/test_task
                def exists_side_effect(path):
                    if path in ["/etc/crontab", "/etc/cron.d/test_task"]:
                        return True
                    return False
                mock_exists.side_effect = exists_side_effect

                with patch("gptscan.Path") as mock_path_cls:
                    mock_cron_d = MagicMock()
                    mock_cron_d.exists.return_value = True
                    mock_cron_d.is_dir.return_value = True

                    mock_file = MagicMock()
                    mock_file.is_file.return_value = True
                    mock_file.__str__.return_value = "/etc/cron.d/test_task"
                    mock_file.name = "test_task"

                    mock_cron_d.iterdir.return_value = [mock_file]

                    def path_side_effect(p):
                        if str(p) == "/etc/cron.d":
                            return mock_cron_d
                        return MagicMock()

                    mock_path_cls.side_effect = path_side_effect

                    # Mock open for /etc/crontab and /etc/cron.d/test_task
                    def open_side_effect(path, *args, **kwargs):
                        if path == "/etc/crontab":
                            return mock_open(read_data=mock_etc_crontab).return_value
                        if path == "/etc/cron.d/test_task":
                            return mock_open(read_data=mock_cron_d_file).return_value
                        raise FileNotFoundError(path)

                    with patch("builtins.open", side_effect=open_side_effect):
                        tasks = get_scheduled_task_commands()

                        # 2 from crontab -l, 1 from /etc/crontab, 2 from /etc/cron.d/test_task = 5
                        assert len(tasks) == 5
                        assert tasks[0] == ("[Cron] User", b"/usr/bin/python3 /path/to/script.py")
                        assert tasks[1] == ("[Cron] User", b"/usr/bin/cleanup.sh")
                        assert tasks[2] == ("[Cron] System (crontab)", b"cd / && run-parts --report /etc/cron.hourly")
                        assert tasks[3] == ("[Cron] System (test_task)", b"/usr/local/bin/daily_backup.sh")
                        assert tasks[4] == ("[Cron] System (test_task)", b"/usr/bin/reboot_task")

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
    with patch("sys.platform", "linux"):
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "crontab")):
            with patch("gptscan.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                with patch("os.path.exists", return_value=False):
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
