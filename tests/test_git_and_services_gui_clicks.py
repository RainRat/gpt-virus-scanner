import sys
import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_scan_git_hooks_click_success(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_hooks_paths", lambda path: ["/mock/repo/.git/hooks/pre-commit"])

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_hooks_click()

    mock_set_target.assert_called_once_with(["/mock/repo/.git/hooks/pre-commit"])
    mock_button_click.assert_called_once()

def test_scan_git_hooks_click_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_hooks_paths", lambda path: [])

    mock_showinfo = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    gptscan.scan_git_hooks_click()

    mock_showinfo.assert_called_once_with("Git Hooks", "No Git hooks found to scan.")

def test_scan_git_hooks_click_error(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    def mock_error(path):
        raise Exception("Failed to scan hooks")
    monkeypatch.setattr(gptscan, "get_git_hooks_paths", mock_error)

    mock_showwarning = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_showwarning)

    gptscan.scan_git_hooks_click()

    mock_showwarning.assert_called_once()
    args, _ = mock_showwarning.call_args
    assert args[0] == "Git Hooks Error"
    assert "Failed to scan hooks" in args[1]

def test_scan_git_stash_click_success(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/stash/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_stash_snippets", lambda path: [("stash@{0}", b"stash data")])

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_stash_click()

    mock_button_click.assert_called_once_with(extra_snippets=[("stash@{0}", b"stash data")])

def test_scan_git_stash_click_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/stash/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_stash_snippets", lambda path: [])

    mock_showinfo = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    gptscan.scan_git_stash_click()

    mock_showinfo.assert_called_once_with("Git Stash", "No Git stashes found to scan.")

def test_scan_git_stash_click_error(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/stash/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    def mock_error(path):
        raise Exception("Stash error")
    monkeypatch.setattr(gptscan, "get_git_stash_snippets", mock_error)

    mock_showwarning = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_showwarning)

    gptscan.scan_git_stash_click()

    mock_showwarning.assert_called_once()
    args, _ = mock_showwarning.call_args
    assert args[0] == "Git Stash Error"
    assert "Stash error" in args[1]

def test_scan_git_conflicts_click_success(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/conflict/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_conflict_snippets", lambda path: [("conflict", b"conflict data")])

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_conflicts_click()

    mock_button_click.assert_called_once_with(extra_snippets=[("conflict", b"conflict data")])

def test_scan_git_conflicts_click_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/conflict/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, "get_git_conflict_snippets", lambda path: [])

    mock_showinfo = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    gptscan.scan_git_conflicts_click()

    mock_showinfo.assert_called_once_with("Git Conflicts", "No Git merge conflicts found to scan.")

def test_scan_git_conflicts_click_error(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/mock/conflict/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    def mock_error(path):
        raise Exception("Conflict error")
    monkeypatch.setattr(gptscan, "get_git_conflict_snippets", mock_error)

    mock_showwarning = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_showwarning)

    gptscan.scan_git_conflicts_click()

    mock_showwarning.assert_called_once()
    args, _ = mock_showwarning.call_args
    assert args[0] == "Git Conflicts Error"
    assert "Conflict error" in args[1]

def test_scan_system_services_click_paths_and_snippets(monkeypatch):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: ["/etc/systemd/system/malware.service"])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [("Service Command", b"exec malicious")])

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_system_services_click()

    mock_set_target.assert_called_once_with(["/etc/systemd/system/malware.service"])
    mock_button_click.assert_called_once_with(extra_snippets=[("Service Command", b"exec malicious")])

def test_scan_system_services_click_paths_only(monkeypatch):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: ["/etc/systemd/system/malware.service"])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [])

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_system_services_click()

    mock_set_target.assert_called_once_with(["/etc/systemd/system/malware.service"])
    mock_button_click.assert_called_once_with(extra_snippets=[])

def test_scan_system_services_click_snippets_only(monkeypatch):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: [])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [("Service Command", b"exec malicious")])

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_system_services_click()

    mock_set_target.assert_not_called()
    mock_button_click.assert_called_once_with(extra_snippets=[("Service Command", b"exec malicious")])

def test_scan_system_services_click_empty(monkeypatch):
    monkeypatch.setattr(gptscan, "get_system_service_paths", lambda: [])
    monkeypatch.setattr(gptscan, "get_system_service_commands", lambda: [])

    mock_showinfo = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    gptscan.scan_system_services_click()

    mock_showinfo.assert_called_once_with("System Services", "No system services were found to scan.")

def test_scan_system_services_click_error(monkeypatch):
    def mock_error():
        raise Exception("Services scan error")
    monkeypatch.setattr(gptscan, "get_system_service_paths", mock_error)

    mock_showwarning = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_showwarning)

    gptscan.scan_system_services_click()

    mock_showwarning.assert_called_once()
    args, _ = mock_showwarning.call_args
    assert args[0] == "System Services Error"
    assert "Services scan error" in args[1]


def test_get_target_path_with_value(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "   /some/path/to/scan   "
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)
    assert gptscan._get_target_path() == "/some/path/to/scan"


def test_get_target_path_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "   "
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)
    assert gptscan._get_target_path() == "."


def test_get_target_path_no_textbox(monkeypatch):
    monkeypatch.setattr(gptscan, "textbox", None, raising=False)
    assert gptscan._get_target_path() == "."
