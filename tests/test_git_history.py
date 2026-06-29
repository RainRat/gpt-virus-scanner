import pytest
from unittest.mock import patch, MagicMock
import gptscan
import subprocess
import os

def test_get_git_history_snippets(monkeypatch):
    # Mock _get_git_info to return a fake toplevel
    monkeypatch.setattr("gptscan._get_git_info", lambda path: ("/fake/repo", "rel/path"))

    def mock_check_output(cmd, cwd=None, **kwargs):
        if "log" in cmd:
            return "commit 1234567890abcdef\nAuthor: User <user@example.com>\nDate:   Mon Jan 1 00:00:00 2024 +0000\n\n    Initial commit\n\ndiff --git a/test.py b/test.py\nnew file mode 100644\nindex 0000000..e69de29\n"
        return ""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    snippets = gptscan.get_git_history_snippets()
    assert len(snippets) == 1
    assert snippets[0][0] == "[Git History]"
    assert b"commit 1234567890abcdef" in snippets[0][1]

def test_unpack_content_git_log():
    content = b"commit abcdef1234567890\nAuthor: User\nDate: Mon Jan 1\n\n    Msg\n\ndiff --git a/script.py b/script.py\n--- a/script.py\n+++ b/script.py\n@@ -1,1 +1,2 @@\n+print('hello')\n+os.system('id')\n"

    snippets = list(gptscan.unpack_content("history", content))

    # It should extract one snippet for script.py with the commit hash
    assert len(snippets) == 1
    name, snippet_content = snippets[0]
    assert "abcdef1" in name
    assert "script.py" in name
    assert b"print('hello')" in snippet_content
    assert b"os.system('id')" in snippet_content

def test_scan_git_history_click(monkeypatch):
    monkeypatch.setattr("gptscan.get_git_history_snippets", lambda path, count: [("[Git History]", b"patch content")])

    # Mock GUI variables
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr("gptscan.textbox", mock_textbox)

    # Mock simpledialog
    monkeypatch.setattr("tkinter.simpledialog.askinteger", lambda title, prompt, **kwargs: 5)

    clicked = False
    def mock_button_click(extra_snippets=None):
        nonlocal clicked
        clicked = True
        assert extra_snippets == [("[Git History]", b"patch content")]

    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_git_history_click()
    assert clicked

def test_cli_git_history_flag(monkeypatch):
    monkeypatch.setattr("gptscan.get_git_history_snippets", lambda path, count: [("[Git History]", b"patch")])

    cli_args = []
    def mock_run_cli(targets, *args, **kwargs):
        # The snippets are passed to run_cli via extra_snippets kwarg
        extra = kwargs.get("extra_snippets", [])
        assert ("[Git History]", b"patch") in extra
        return 0

    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    import sys
    test_args = ["gptscan.py", "--git-history", "3", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

def test_system_audit_includes_git_history(monkeypatch):
    # Mock all discovery functions except git history
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: [])
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
    monkeypatch.setattr("gptscan.get_git_stash_snippets", lambda: [])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_nodejs_package_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_ruby_gems_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_php_packages_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_rust_packages_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_go_packages_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_java_packages_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_dotnet_packages_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_editor_extensions_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_documents_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_downloads_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_desktop_paths", lambda: [])
    monkeypatch.setattr("gptscan.get_temp_paths", lambda: [])

    # Mock git history to return something
    monkeypatch.setattr("gptscan.get_git_history_snippets", lambda: [("[Git History]", b"audit patch")])

    paths, snippets = gptscan.get_system_audit_data()
    assert ("[Git History]", b"audit patch") in snippets
