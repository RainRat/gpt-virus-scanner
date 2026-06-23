import pytest
from unittest.mock import patch, MagicMock
import gptscan
import os
import sys

def test_new_discovery_functions(monkeypatch):
    # Mock os.path.isdir to return True for our test paths
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    # Mock Path.exists to return True
    monkeypatch.setattr("gptscan.Path.exists", lambda x: True)

    # Test get_documents_paths
    with patch("gptscan.Path.home", return_value=gptscan.Path("/home/user")):
        paths = gptscan.get_documents_paths()
        assert str(gptscan.Path("/home/user/Documents")) in paths

    # Test get_ruby_gems_paths
    monkeypatch.setenv("GEM_HOME", "/gem/home")
    paths = gptscan.get_ruby_gems_paths()
    assert "/gem/home/gems" in paths

    # Test get_php_packages_paths
    def mock_check_output(cmd, **kwargs):
        if cmd[0] == 'composer':
            return "/composer/home"
        return ""
    monkeypatch.setattr("subprocess.check_output", mock_check_output)
    paths = gptscan.get_php_packages_paths()
    assert "/composer/home/vendor" in paths

    # Test get_rust_packages_paths
    monkeypatch.setenv("CARGO_HOME", "/cargo/home")
    with patch("glob.glob", return_value=["/cargo/home/registry/src/github.com-1ecc6299db9ec823"]):
        paths = gptscan.get_rust_packages_paths()
        assert "/cargo/home/registry/src/github.com-1ecc6299db9ec823" in paths

    # Test get_go_packages_paths
    monkeypatch.setenv("GOPATH", "/go/path")
    paths = gptscan.get_go_packages_paths()
    assert "/go/path/pkg/mod" in paths

def test_system_audit_integration(monkeypatch):
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_shell_history_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_path_directories", lambda: [])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_nodejs_package_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_editor_extensions_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_downloads_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_desktop_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_temp_paths", lambda: [])

    # New ones
    monkeypatch.setattr("gptscan.get_ruby_gems_paths", lambda: ["/ruby/gems"])
    monkeypatch.setattr("gptscan.get_php_packages_paths", lambda: ["/php/packages"])
    monkeypatch.setattr("gptscan.get_rust_packages_paths", lambda: ["/rust/packages"])
    monkeypatch.setattr("gptscan.get_go_packages_paths", lambda: ["/go/packages"])
    monkeypatch.setattr("gptscan.get_documents_paths", lambda: ["/user/documents"])

    monkeypatch.setattr("gptscan.get_running_process_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_environment_variable_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_scheduled_task_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_startup_item_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_system_service_commands", lambda: [])
    monkeypatch.setattr("gptscan.get_git_config_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_git_stash_snippets", lambda: [])

    paths, snippets = gptscan.get_system_audit_data()
    assert "/ruby/gems" in paths
    assert "/php/packages" in paths
    assert "/rust/packages" in paths
    assert "/go/packages" in paths
    assert "/user/documents" in paths

def test_cli_new_flags(monkeypatch):
    captured_targets = []
    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_targets
        captured_targets = targets
        return 0

    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)
    monkeypatch.setattr("gptscan.get_documents_paths", lambda: ["/docs"])
    monkeypatch.setattr("gptscan.get_ruby_gems_paths", lambda: ["/gems"])

    test_args = ["gptscan.py", "--documents", "--ruby-gems", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

    assert "/docs" in captured_targets
    assert "/gems" in captured_targets
