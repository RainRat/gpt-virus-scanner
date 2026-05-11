import pytest
from unittest.mock import patch, MagicMock
import gptscan
from pathlib import Path

def test_get_ssh_config_paths(mocker):
    home = Path("/home/user")
    mocker.patch("pathlib.Path.home", return_value=home)

    # Mock Path object and its exists method
    mock_exists = mocker.patch("pathlib.Path.exists")
    # Make everything exist
    mock_exists.return_value = True

    paths = gptscan.get_ssh_config_paths()
    assert "/home/user/.ssh/config" in paths
    assert "/home/user/.ssh/authorized_keys" in paths
    assert "/etc/ssh/sshd_config" in paths
    assert "/etc/ssh/ssh_config" in paths

def test_scan_system_audit_click(mocker):
    mocker.patch("gptscan.get_shell_profile_paths", return_value=["/p1"])
    mocker.patch("gptscan.get_shell_history_paths", return_value=["/h1"])
    mocker.patch("gptscan.get_system_path_directories", return_value=["/bin"])
    mocker.patch("gptscan.get_ssh_config_paths", return_value=["/s1"])
    mocker.patch("gptscan.get_running_process_commands", return_value=[("proc", b"cmd")])
    mocker.patch("gptscan.get_environment_variable_snippets", return_value=[("env", b"val")])
    mocker.patch("gptscan.get_scheduled_task_commands", return_value=[("task", b"run")])
    mocker.patch("gptscan.get_startup_item_commands", return_value=[("start", b"up")])

    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")

    gptscan.scan_system_audit_click()

    mock_set_target.assert_called_once()
    args, _ = mock_set_target.call_args
    assert "/p1" in args[0]
    assert "/h1" in args[0]
    assert "/bin" in args[0]
    assert "/s1" in args[0]

    mock_button_click.assert_called_once()
    _, kwargs = mock_button_click.call_args
    snippets = kwargs["extra_snippets"]
    assert len(snippets) == 4

def test_cli_audit_flag(mocker):
    mocker.patch("gptscan.get_shell_profile_paths", return_value=["/p1"])
    mocker.patch("gptscan.get_shell_history_paths", return_value=[])
    mocker.patch("gptscan.get_system_path_directories", return_value=[])
    mocker.patch("gptscan.get_ssh_config_paths", return_value=[])
    mocker.patch("gptscan.get_running_process_commands", return_value=[])
    mocker.patch("gptscan.get_environment_variable_snippets", return_value=[])
    mocker.patch("gptscan.get_scheduled_task_commands", return_value=[])
    mocker.patch("gptscan.get_startup_item_commands", return_value=[])

    mock_run_cli = mocker.patch("gptscan.run_cli", return_value=0)

    import sys
    test_args = ["gptscan.py", "--audit", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

    mock_run_cli.assert_called_once()
    args, _ = mock_run_cli.call_args
    assert "/p1" in args[0]
