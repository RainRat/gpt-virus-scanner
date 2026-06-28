import pytest
from unittest.mock import patch, MagicMock, mock_open
import gptscan

def test_get_scheduled_task_commands_system_crontab_robustness():
    """Test that system crontab parsing correctly handles commands with equals signs and skips env vars."""
    mock_etc_crontab = """
# /etc/crontab: system-wide crontab

SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
VAR_WITH_SPACE = value

# m h dom mon dow user  command
17 *    * * *   root    cd / && run-parts --report /etc/cron.hourly
25 6    * * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )
@reboot         root    /usr/bin/python3 /opt/app.py --arg=val
"""
    with patch("sys.platform", "linux"), \
         patch("os.path.exists", side_effect=lambda p: p == "/etc/crontab"), \
         patch("gptscan.Path") as mock_path_cls:

        mock_cron_d = MagicMock()
        mock_cron_d.exists.return_value = False
        mock_path_cls.return_value = mock_cron_d

        with patch("builtins.open", mock_open(read_data=mock_etc_crontab)):
            tasks = gptscan.get_scheduled_task_commands()

            # Should have 3 commands from the crontab
            # 1. cd / && run-parts --report /etc/cron.hourly
            # 2. test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )
            # 3. /usr/bin/python3 /opt/app.py --arg=val

            assert len(tasks) == 3
            commands = [t[1] for t in tasks]
            assert b"run-parts --report /etc/cron.hourly" in commands[0]
            assert b"anacron" in commands[1]
            assert b"--arg=val" in commands[2]

            # Ensure env vars were NOT captured as commands
            assert b"SHELL=/bin/sh" not in commands
            assert b"PATH=" not in [c[:5] for c in commands]
            assert b"VAR_WITH_SPACE" not in commands

def test_get_scheduled_task_commands_cron_d_robustness():
    """Test that /etc/cron.d parsing also benefits from the fix."""
    mock_cron_d_file = """
# A file in /etc/cron.d/
CUSTOM_VAR=123
* * * * * user /usr/bin/app --config=/etc/app.conf
@daily    user /usr/bin/cleanup --force=true
"""
    with patch("sys.platform", "linux"), \
         patch("os.path.exists", side_effect=lambda p: p == "/etc/cron.d/test_task"), \
         patch("gptscan.Path") as mock_path_cls:

        mock_cron_d = MagicMock()
        mock_cron_d.exists.return_value = True
        mock_cron_d.is_dir.return_value = True

        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.__str__.return_value = "/etc/cron.d/test_task"
        mock_file.name = "test_task"
        mock_cron_d.iterdir.return_value = [mock_file]

        # We need mock_path_cls to return mock_cron_d when called with "/etc/cron.d"
        mock_path_cls.side_effect = lambda p: mock_cron_d if str(p) == "/etc/cron.d" else MagicMock()

        with patch("builtins.open", mock_open(read_data=mock_cron_d_file)):
            tasks = gptscan.get_scheduled_task_commands()

            assert len(tasks) == 2
            commands = [t[1] for t in tasks]
            assert b"--config=/etc/app.conf" in commands[0]
            assert b"--force=true" in commands[1]
            assert b"CUSTOM_VAR" not in commands
