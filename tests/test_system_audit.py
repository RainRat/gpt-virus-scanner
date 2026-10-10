from unittest.mock import MagicMock
import gptscan

def test_get_system_audit_data_aggregation(monkeypatch):
    """Verify that get_system_audit_data aggregates paths and virtual snippets from all target gatherers."""
    monkeypatch.setattr(gptscan, "get_shell_profile_paths", lambda: ["/mock/profile"])
    monkeypatch.setattr(gptscan, "get_shell_history_paths", lambda: ["/mock/history"])
    monkeypatch.setattr(gptscan, "get_system_path_directories", lambda: ["/mock/syspath"])
    monkeypatch.setattr(gptscan, "get_ssh_config_paths", lambda: ["/mock/ssh"])
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: ["/mock/services"])
    monkeypatch.setattr(gptscan, "get_network_config_paths", lambda: ["/mock/network"])
    monkeypatch.setattr(gptscan, "get_git_hooks_paths", lambda: ["/mock/githooks"])
    monkeypatch.setattr(gptscan, "get_python_package_paths", lambda: ["/mock/python"])
    monkeypatch.setattr(gptscan, "get_nodejs_package_paths", lambda: ["/mock/nodejs"])
    monkeypatch.setattr(gptscan, "get_ruby_gems_paths", lambda: ["/mock/ruby"])
    monkeypatch.setattr(gptscan, "get_php_packages_paths", lambda: ["/mock/php"])
    monkeypatch.setattr(gptscan, "get_rust_packages_paths", lambda: ["/mock/rust"])
    monkeypatch.setattr(gptscan, "get_go_packages_paths", lambda: ["/mock/go"])
    monkeypatch.setattr(gptscan, "get_java_packages_paths", lambda: ["/mock/java"])
    monkeypatch.setattr(gptscan, "get_dotnet_packages_paths", lambda: ["/mock/dotnet"])
    monkeypatch.setattr(gptscan, "get_browser_extensions_paths", lambda: ["/mock/browser_ext"])
    monkeypatch.setattr(gptscan, "get_editor_extensions_paths", lambda: ["/mock/editor_ext"])
    monkeypatch.setattr(gptscan, "get_documents_paths", lambda: ["/mock/docs"])
    monkeypatch.setattr(gptscan, "get_downloads_paths", lambda: ["/mock/downloads"])
    monkeypatch.setattr(gptscan, "get_desktop_paths", lambda: ["/mock/desktop"])
    monkeypatch.setattr(gptscan, "get_temp_paths", lambda: ["/mock/temp"])
    monkeypatch.setattr(gptscan, "get_env_file_paths", lambda: ["/mock/env"])

    monkeypatch.setattr(gptscan, "get_running_process_commands", lambda: [("Process: ps", b"ps aux")])
    monkeypatch.setattr(gptscan, "get_environment_variable_snippets", lambda: [("Env: PATH", b"PATH=/usr/bin")])
    monkeypatch.setattr(gptscan, "get_scheduled_task_commands", lambda: [("Task: cron", b"* * * * *")])
    monkeypatch.setattr(gptscan, "get_startup_item_commands", lambda: [("Startup: init", b"exec init")])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [("Service: nginx", b"nginx.service")])
    monkeypatch.setattr(gptscan, "get_git_config_snippets", lambda: [("GitConfig: user.name", b"name = test")])
    monkeypatch.setattr(gptscan, "get_git_stash_snippets", lambda: [("GitStash: stash@{0}", b"stash content")])
    monkeypatch.setattr(gptscan, "get_browser_bookmarks_snippets", lambda: [("Bookmark: example.com", b"http://example.com")])

    paths, snippets = gptscan.get_system_audit_data()

    assert "/mock/profile" in paths
    assert "/mock/history" in paths
    assert "/mock/syspath" in paths
    assert "/mock/ssh" in paths
    assert "/mock/services" in paths
    assert "/mock/network" in paths
    assert "/mock/githooks" in paths
    assert "/mock/python" in paths
    assert "/mock/nodejs" in paths
    assert "/mock/ruby" in paths
    assert "/mock/php" in paths
    assert "/mock/rust" in paths
    assert "/mock/go" in paths
    assert "/mock/java" in paths
    assert "/mock/dotnet" in paths
    assert "/mock/browser_ext" in paths
    assert "/mock/editor_ext" in paths
    assert "/mock/docs" in paths
    assert "/mock/downloads" in paths
    assert "/mock/desktop" in paths
    assert "/mock/temp" in paths
    assert "/mock/env" in paths

    assert ("Process: ps", b"ps aux") in snippets
    assert ("Env: PATH", b"PATH=/usr/bin") in snippets
    assert ("Task: cron", b"* * * * *") in snippets
    assert ("Startup: init", b"exec init") in snippets
    assert ("Service: nginx", b"nginx.service") in snippets
    assert ("GitConfig: user.name", b"name = test") in snippets
    assert ("GitStash: stash@{0}", b"stash content") in snippets
    assert ("Bookmark: example.com", b"http://example.com") in snippets

def test_scan_system_audit_click(monkeypatch):
    """Verify that scan_system_audit_click calls _generic_scan_click with get_system_audit_data."""
    mock_generic_click = MagicMock()
    monkeypatch.setattr(gptscan, "_generic_scan_click", mock_generic_click)

    gptscan.scan_system_audit_click()

    mock_generic_click.assert_called_once_with(
        gptscan.get_system_audit_data,
        "System Audit",
        "No system items were found to scan.",
        "System Audit Error"
    )
