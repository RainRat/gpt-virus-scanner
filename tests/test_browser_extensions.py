import sys
import os
from pathlib import Path
import pytest
import gptscan
from gptscan import get_browser_extensions_paths, scan_browser_extensions_click

def test_get_browser_extensions_paths_linux(monkeypatch):
    """Verify the logic of get_browser_extensions_paths on Linux."""
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(sys, "platform", "linux")

    fake_paths = [
        home / ".config" / "google-chrome" / "Default" / "Extensions",
        home / ".mozilla" / "firefox" / "profile.default" / "extensions",
    ]

    def mock_exists(self):
        return self in fake_paths or str(self).endswith(".mozilla/firefox")

    def mock_isdir(p):
        return Path(p) in fake_paths

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    def mock_glob(self, pattern):
        if str(self).endswith(".mozilla/firefox") and pattern == "*/extensions":
            return [home / ".mozilla" / "firefox" / "profile.default" / "extensions"]
        return []

    monkeypatch.setattr(Path, "glob", mock_glob)

    monkeypatch.setattr(gptscan, "_normalize_and_filter_dirs", lambda paths: [str(p) for p in paths if p])

    paths = get_browser_extensions_paths()

    assert str(home / ".config" / "google-chrome" / "Default" / "Extensions") in paths
    assert str(home / ".mozilla" / "firefox" / "profile.default" / "extensions") in paths

def test_get_browser_extensions_paths_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "/Users/user/AppData/Local")
    monkeypatch.setenv("APPDATA", "/Users/user/AppData/Roaming")

    def mock_glob(self, pattern):
        s = str(self).replace("\\", "/")
        if "Google/Chrome/User Data" in s and pattern == "Profile */Extensions":
            return [Path("/Users/user/AppData/Local/Google/Chrome/User Data/Profile 1/Extensions")]
        if "Microsoft/Edge/User Data" in s and pattern == "Profile */Extensions":
            return [Path("/Users/user/AppData/Local/Microsoft/Edge/User Data/Profile 2/Extensions")]
        if "Mozilla/Firefox/Profiles" in s and pattern == "*/extensions":
            return [Path("/Users/user/AppData/Roaming/Mozilla/Firefox/Profiles/xyz.default/extensions")]
        return []

    monkeypatch.setattr(Path, "glob", mock_glob)
    monkeypatch.setattr(gptscan, "_normalize_and_filter_dirs", lambda paths: [str(p) for p in paths if p])

    paths = get_browser_extensions_paths()

    assert str(Path("/Users/user/AppData/Local/Google/Chrome/User Data/Default/Extensions")) in paths
    assert str(Path("/Users/user/AppData/Local/Google/Chrome/User Data/Profile 1/Extensions")) in paths
    assert str(Path("/Users/user/AppData/Local/Microsoft/Edge/User Data/Default/Extensions")) in paths
    assert str(Path("/Users/user/AppData/Local/Microsoft/Edge/User Data/Profile 2/Extensions")) in paths
    assert str(Path("/Users/user/AppData/Roaming/Mozilla/Firefox/Profiles/xyz.default/extensions")) in paths

def test_get_browser_extensions_paths_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    home = Path("/Users/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    def mock_glob(self, pattern):
        s = str(self).replace("\\", "/")
        if "Google/Chrome" in s and pattern == "Profile */Extensions":
            return [home / "Library" / "Application Support" / "Google" / "Chrome" / "Profile 1" / "Extensions"]
        if "Microsoft Edge" in s and pattern == "Profile */Extensions":
            return [home / "Library" / "Application Support" / "Microsoft Edge" / "Profile 2" / "Extensions"]
        if "Firefox/Profiles" in s and pattern == "*/extensions":
            return [home / "Library" / "Application Support" / "Firefox" / "Profiles" / "abc.default" / "extensions"]
        return []

    monkeypatch.setattr(Path, "glob", mock_glob)
    monkeypatch.setattr(gptscan, "_normalize_and_filter_dirs", lambda paths: [str(p) for p in paths if p])

    paths = get_browser_extensions_paths()

    assert str(home / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "Extensions") in paths
    assert str(home / "Library" / "Application Support" / "Google" / "Chrome" / "Profile 1" / "Extensions") in paths
    assert str(home / "Library" / "Application Support" / "Microsoft Edge" / "Default" / "Extensions") in paths
    assert str(home / "Library" / "Application Support" / "Microsoft Edge" / "Profile 2" / "Extensions") in paths
    assert str(home / "Library" / "Application Support" / "Firefox" / "Profiles" / "abc.default" / "extensions") in paths

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

def test_scan_browser_extensions_click_no_paths(monkeypatch):
    """Verify the GUI callback when no paths are found."""
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: [])

    message_box_shown = False
    def mock_showinfo(title, message):
        nonlocal message_box_shown
        message_box_shown = True
        assert "No browser extension folders" in message

    import gptscan
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    scan_browser_extensions_click()
    assert message_box_shown
