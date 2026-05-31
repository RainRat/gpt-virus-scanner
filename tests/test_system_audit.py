import pytest
from unittest.mock import patch, MagicMock
import gptscan
from pathlib import Path

def test_scan_system_audit_click(monkeypatch):
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: ["/p1"])
    monkeypatch.setattr("gptscan.get_shell_history_paths", lambda: ["/h1"])
    monkeypatch.setattr("gptscan.get_system_path_directories", lambda: ["/bin"])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", lambda: ["/s1"])
    monkeypatch.setattr("gptscan.get_running_process_commands", lambda: [("proc", b"cmd")])
    monkeypatch.setattr("gptscan.get_environment_variable_snippets", lambda: [("env", b"val")])
    monkeypatch.setattr("gptscan.get_scheduled_task_commands", lambda: [("task", b"run")])
    monkeypatch.setattr("gptscan.get_startup_item_commands", lambda: [("start", b"up")])
    monkeypatch.setattr("gptscan.get_system_service_paths", lambda: ["/ser1"])
    monkeypatch.setattr("gptscan.get_system_service_commands", lambda: [("ser", b"cmd")])
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda: ["/hook1"])
    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [("git", b"conf")])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: ["/pkg1"])
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: ["/browser1"])

    target_paths = []
    def mock_set_target(paths):
        nonlocal target_paths
        target_paths = paths

    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)

    clicked = False
    snippets_count = 0
    def mock_button_click(extra_snippets=None, **kwargs):
        nonlocal clicked, snippets_count
        clicked = True
        if extra_snippets:
            snippets_count = len(extra_snippets)

    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_system_audit_click()

    assert "/p1" in target_paths
    assert "/h1" in target_paths
    assert "/bin" in target_paths
    assert "/s1" in target_paths
    assert "/ser1" in target_paths
    assert "/hook1" in target_paths
    assert "/pkg1" in target_paths
    assert "/browser1" in target_paths

    assert clicked
    assert snippets_count == 6

def test_cli_audit_flag(monkeypatch):
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: ["/p1"])
    monkeypatch.setattr("gptscan.get_shell_history_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_path_directories", lambda: [])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_running_process_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_environment_variable_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_scheduled_task_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_startup_item_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: ["/browser-cli"])

    cli_args = []
    def mock_run_cli(targets, *args, **kwargs):
        nonlocal cli_args
        cli_args = targets
        return 0

    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    import sys
    test_args = ["gptscan.py", "--audit", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

    assert "/p1" in cli_args
    assert "/browser-cli" in cli_args
