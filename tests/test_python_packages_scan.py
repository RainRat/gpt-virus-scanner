import sys
import os
import pytest
from gptscan import get_python_package_paths, scan_python_packages_click, Config
import tkinter as tk

def test_get_python_package_paths():
    """Verify that get_python_package_paths returns existing site-packages directories."""
    paths = get_python_package_paths()
    assert isinstance(paths, list)
    # On most systems running tests, there should be at least one site-packages dir
    # If not (e.g. very minimal environment), this might be empty, but usually it's not.
    if paths:
        for p in paths:
            assert os.path.isdir(p)
            assert 'site-packages' in p.lower() or 'dist-packages' in p.lower() or 'lib' in p.lower()
            assert os.path.isabs(p)

def test_get_python_package_paths_mocked(monkeypatch):
    """Verify the logic of get_python_package_paths with mocked inputs."""
    def mock_getsitepackages():
        return ["/fake/site-packages", "/another/fake/path"]

    def mock_getusersitepackages():
        return "/user/fake/site-packages"

    import site
    monkeypatch.setattr(site, "getsitepackages", mock_getsitepackages, raising=False)
    monkeypatch.setattr(site, "getusersitepackages", mock_getusersitepackages, raising=False)
    monkeypatch.setattr(sys, "path", ["/env/site-packages", "/random/path"])

    # Mock os.path.isdir to return True for our fake paths
    def mock_isdir(p):
        return "fake" in p or "site-packages" in p
    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    paths = get_python_package_paths()
    assert "/fake/site-packages" in paths
    assert "/another/fake/path" in paths
    assert "/user/fake/site-packages" in paths
    assert "/env/site-packages" in paths
    assert "/random/path" not in paths # Should be filtered out as it doesn't contain 'site-packages' and is not from site.get*

def test_scan_python_packages_click(monkeypatch):
    """Verify the GUI callback for scanning Python packages."""
    target_paths = []
    def mock_set_scan_target(paths):
        nonlocal target_paths
        target_paths = paths

    def mock_button_click():
        pass

    monkeypatch.setattr("gptscan._set_scan_target", mock_set_scan_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: ["/mock/site-packages"])

    scan_python_packages_click()
    assert target_paths == ["/mock/site-packages"]

def test_scan_python_packages_click_no_paths(monkeypatch):
    """Verify the GUI callback when no paths are found."""
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: [])

    message_box_shown = False
    def mock_showinfo(title, message):
        nonlocal message_box_shown
        message_box_shown = True
        assert "No Python site-packages" in message

    import gptscan
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    scan_python_packages_click()
    assert message_box_shown
