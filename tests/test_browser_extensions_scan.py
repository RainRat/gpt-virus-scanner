import sys
import os
from pathlib import Path
import pytest
from gptscan import get_browser_extensions_paths, scan_browser_extensions_click

def test_get_browser_extensions_paths_linux(monkeypatch):
    """Verify browser extension discovery on Linux."""
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(sys, "platform", "linux")

    fake_paths = [
        home / ".config" / "google-chrome" / "Default" / "Extensions",
        home / ".mozilla" / "firefox" / "abcdef.default-release" / "extensions",
    ]

    def mock_glob(self, pattern):
        if pattern == "google-chrome/*/Extensions":
            return [home / ".config" / "google-chrome" / "Default" / "Extensions"]
        if pattern == "chromium/*/Extensions":
            return []
        if pattern == ".mozilla/firefox/*/extensions":
            return [home / ".mozilla" / "firefox" / "abcdef.default-release" / "extensions"]
        return []

    def mock_isdir(p):
        return Path(p) in fake_paths

    monkeypatch.setattr(Path, "glob", mock_glob)
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    # Ensure abspath doesn't change our /home/user paths significantly
    monkeypatch.setattr(os.path, "abspath", lambda x: x)

    paths = get_browser_extensions_paths()
    assert str(home / ".config" / "google-chrome" / "Default" / "Extensions") in paths
    assert str(home / ".mozilla" / "firefox" / "abcdef.default-release" / "extensions") in paths

def test_get_browser_extensions_paths_windows(monkeypatch):
    """Verify browser extension discovery on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")
    local_appdata = "/mock/localappdata"
    appdata = "/mock/appdata"
    monkeypatch.setenv("LOCALAPPDATA", local_appdata)
    monkeypatch.setenv("APPDATA", appdata)

    fake_paths = [
        Path(local_appdata) / "Google" / "Chrome" / "User Data" / "Default" / "Extensions",
        Path(appdata) / "Mozilla" / "Firefox" / "Profiles" / "xyz.default" / "extensions",
    ]

    def mock_glob(self, pattern):
        if str(self) == local_appdata:
            if pattern == "Google/Chrome/User Data/*/Extensions":
                return [Path(local_appdata) / "Google" / "Chrome" / "User Data" / "Default" / "Extensions"]
        if str(self) == appdata:
            if pattern == "Mozilla/Firefox/Profiles/*/extensions":
                return [Path(appdata) / "Mozilla" / "Firefox" / "Profiles" / "xyz.default" / "extensions"]
        return []

    def mock_isdir(p):
        return Path(p) in fake_paths

    monkeypatch.setattr(Path, "glob", mock_glob)
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    monkeypatch.setattr(os.path, "abspath", lambda x: x)

    paths = get_browser_extensions_paths()
    assert str(Path(local_appdata) / "Google" / "Chrome" / "User Data" / "Default" / "Extensions") in paths
    assert str(Path(appdata) / "Mozilla" / "Firefox" / "Profiles" / "xyz.default" / "extensions") in paths

def test_scan_browser_extensions_click(monkeypatch):
    """Verify the GUI callback for scanning browser extensions."""
    target_paths = []
    def mock_set_scan_target(paths):
        nonlocal target_paths
        target_paths = paths

    def mock_button_click():
        pass

    monkeypatch.setattr("gptscan._set_scan_target", mock_set_scan_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: ["/mock/chrome/extensions"])

    scan_browser_extensions_click()
    assert target_paths == ["/mock/chrome/extensions"]
