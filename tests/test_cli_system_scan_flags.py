import sys
from unittest.mock import MagicMock
import pytest
import gptscan


@pytest.mark.parametrize(
    "flag,func_name,mock_val,is_snippet",
    [
        ("--shell-profiles", "get_shell_profile_paths", ["/mock/profile"], False),
        ("--shell-history", "get_shell_history_paths", ["/mock/history"], False),
        ("--running-processes", "get_running_process_commands", [("[Proc]", b"cmd")], True),
        ("--scheduled-tasks", "get_scheduled_task_commands", [("[Task]", b"cron")], True),
        ("--startup-items", "get_startup_item_commands", [("[Startup]", b"item")], True),
        ("--python-packages", "get_python_package_paths", ["/mock/site-packages"], False),
        ("--browser-bookmarks", "get_browser_bookmarks_snippets", [("[Bookmark]", b"javascript:...")], True),
        ("--nodejs-packages", "get_nodejs_package_paths", ["/mock/node_modules"], False),
        ("--browser-extensions", "get_browser_extensions_paths", ["/mock/extensions"], False),
        ("--editor-extensions", "get_editor_extensions_paths", ["/mock/vscode"], False),
        ("--ssh-config", "get_ssh_config_paths", ["/mock/.ssh/config"], False),
        ("--network-config", "get_network_config_paths", ["/mock/hosts"], False),
        ("--ruby-gems", "get_ruby_gems_paths", ["/mock/gems"], False),
        ("--php-packages", "get_php_packages_paths", ["/mock/vendor"], False),
        ("--rust-packages", "get_rust_packages_paths", ["/mock/cargo"], False),
        ("--go-packages", "get_go_packages_paths", ["/mock/gopath"], False),
        ("--java-packages", "get_java_packages_paths", ["/mock/m2"], False),
        ("--dotnet-packages", "get_dotnet_packages_paths", ["/mock/nuget"], False),
        ("--documents", "get_documents_paths", ["/mock/Documents"], False),
        ("--env-files", "get_env_file_paths", ["/mock/.env"], False),
        ("--desktop", "get_desktop_paths", ["/mock/Desktop"], False),
        ("--temp", "get_temp_paths", ["/mock/tmp"], False),
    ],
)
def test_cli_system_scan_flags_found(monkeypatch, flag, func_name, mock_val, is_snippet):
    monkeypatch.setattr(gptscan, func_name, lambda: mock_val)

    captured_targets = []
    captured_snippets = []

    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_targets, captured_snippets
        captured_targets = targets
        captured_snippets = kwargs.get("extra_snippets", [])
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr("sys.argv", ["gptscan.py", flag, "--cli"])

    gptscan.main()

    if is_snippet:
        assert mock_val[0] in captured_snippets
    else:
        assert mock_val[0] in captured_targets


def test_cli_system_services_flag_found(monkeypatch):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: ["/mock/service.service"])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [("[ServiceCmd]", b"exec")])

    captured_targets = []
    captured_snippets = []

    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_targets, captured_snippets
        captured_targets = targets
        captured_snippets = kwargs.get("extra_snippets", [])
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--system-services", "--cli"])

    gptscan.main()

    assert "/mock/service.service" in captured_targets
    assert ("[ServiceCmd]", b"exec") in captured_snippets


def test_cli_system_services_flag_not_found_warning(monkeypatch, capsys):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: [])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [])

    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--system-services", "--cli"])

    gptscan.main()

    captured = capsys.readouterr()
    assert "No system services or systemd units were found." in captured.err


@pytest.mark.parametrize(
    "flag,func_name,expected_warning",
    [
        ("--shell-profiles", "get_shell_profile_paths", "No common shell profile files were found on this system."),
        ("--shell-history", "get_shell_history_paths", "No common shell history files were found on this system."),
        ("--running-processes", "get_running_process_commands", "No running processes with command lines were found."),
        ("--scheduled-tasks", "get_scheduled_task_commands", "No scheduled tasks or Cron jobs were found."),
        ("--startup-items", "get_startup_item_commands", "No system startup items or LaunchAgents were found."),
        ("--python-packages", "get_python_package_paths", "No Python site-packages folders were found."),
        ("--browser-bookmarks", "get_browser_bookmarks_snippets", "No suspicious browser bookmarklets were found."),
        ("--nodejs-packages", "get_nodejs_package_paths", "No global Node.js package folders were found."),
        ("--browser-extensions", "get_browser_extensions_paths", "No browser extension folders were found."),
        ("--editor-extensions", "get_editor_extensions_paths", "No editor extension folders were found."),
        ("--ssh-config", "get_ssh_config_paths", "No SSH configuration or authorized_keys files were found."),
        ("--network-config", "get_network_config_paths", "No network configuration files were found."),
        ("--ruby-gems", "get_ruby_gems_paths", "No Ruby gems folders were found."),
        ("--php-packages", "get_php_packages_paths", "No global PHP package folders were found."),
        ("--rust-packages", "get_rust_packages_paths", "No global Rust package folders were found."),
        ("--go-packages", "get_go_packages_paths", "No Go package folders were found."),
        ("--java-packages", "get_java_packages_paths", "No Java package folders were found."),
        ("--dotnet-packages", "get_dotnet_packages_paths", "No .NET NuGet package folders were found."),
        ("--env-files", "get_env_file_paths", "No common .env files were found."),
    ],
)
def test_cli_system_scan_flags_not_found_warning_and_quiet_suppression(monkeypatch, capsys, flag, func_name, expected_warning):
    monkeypatch.setattr(gptscan, func_name, lambda: [])

    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)

    # 1. Without --quiet, prints warning to stderr
    monkeypatch.setattr("sys.argv", ["gptscan.py", flag, "--cli"])
    gptscan.main()
    captured = capsys.readouterr()
    assert expected_warning in captured.err

    # 2. With --quiet, suppresses warning
    monkeypatch.setattr("sys.argv", ["gptscan.py", flag, "--quiet", "--cli"])
    gptscan.main()
    captured_quiet = capsys.readouterr()
    assert captured_quiet.err == ""
