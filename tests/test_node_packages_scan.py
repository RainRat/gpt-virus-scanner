import os
import subprocess
import pytest
from gptscan import get_node_package_paths, scan_node_packages_click

def test_get_node_package_paths_mocked(monkeypatch):
    """Verify the logic of get_node_package_paths with mocked inputs."""

    def mock_check_output(cmd, **kwargs):
        if cmd == ["npm", "root", "-g"]:
            return "/fake/global/node_modules\n"
        return ""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # Mock os.path.isdir to return True for our fake paths
    def mock_isdir(p):
        return "node_modules" in p
    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    # Mock os.getcwd
    monkeypatch.setattr(os, "getcwd", lambda: "/current/project")

    paths = get_node_package_paths()

    assert "/fake/global/node_modules" in paths
    assert "/current/project/node_modules" in paths
    assert len(paths) == 2

def test_get_node_package_paths_no_npm(monkeypatch):
    """Verify logic when npm is not found or fails."""

    def mock_check_output(cmd, **kwargs):
        raise FileNotFoundError("npm not found")

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    def mock_isdir(p):
        return "node_modules" in p
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    monkeypatch.setattr(os, "getcwd", lambda: "/current/project")

    paths = get_node_package_paths()

    assert "/current/project/node_modules" in paths
    assert len(paths) == 1

def test_scan_node_packages_click(monkeypatch):
    """Verify the GUI callback for scanning Node.js packages."""
    target_paths = []
    def mock_set_scan_target(paths):
        nonlocal target_paths
        target_paths = paths

    def mock_button_click():
        pass

    monkeypatch.setattr("gptscan._set_scan_target", mock_set_scan_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)
    monkeypatch.setattr("gptscan.get_node_package_paths", lambda: ["/mock/node_modules"])

    scan_node_packages_click()
    assert target_paths == ["/mock/node_modules"]

def test_scan_node_packages_click_no_paths(monkeypatch):
    """Verify the GUI callback when no paths are found."""
    monkeypatch.setattr("gptscan.get_node_package_paths", lambda: [])

    message_box_shown = False
    def mock_showinfo(title, message):
        nonlocal message_box_shown
        message_box_shown = True
        assert "No Node.js node_modules" in message

    import tkinter.messagebox
    monkeypatch.setattr(tkinter.messagebox, "showinfo", mock_showinfo)

    scan_node_packages_click()
    assert message_box_shown
