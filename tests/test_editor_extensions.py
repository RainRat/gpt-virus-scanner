import sys
import os
from pathlib import Path
import pytest
from gptscan import get_editor_extensions_paths, scan_editor_extensions_click

def test_get_editor_extensions_paths_linux(monkeypatch):
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    fake_paths = [
        home / ".vscode" / "extensions",
        home / ".config" / "sublime-text" / "Packages",
        home / ".vim" / "pack",
        home / ".local" / "share" / "nvim" / "site" / "pack",
    ]

    def mock_exists(self):
        return self in fake_paths

    def mock_isdir(p):
        return Path(p) in fake_paths

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    monkeypatch.setattr(sys, "platform", "linux")

    paths = get_editor_extensions_paths()

    assert str(home / ".vscode" / "extensions") in paths
    assert str(home / ".config" / "sublime-text" / "Packages") in paths
    assert str(home / ".vim" / "pack") in paths
    assert str(home / ".local" / "share" / "nvim" / "site" / "pack") in paths

def test_get_editor_extensions_paths_win32_with_env(monkeypatch):
    home = Path("C:/Users/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(sys, "platform", "win32")
    appdata = "C:\\AppData"
    local_appdata = "C:\\LocalAppData"
    monkeypatch.setenv("APPDATA", appdata)
    monkeypatch.setenv("LOCALAPPDATA", local_appdata)

    sublime1 = os.path.abspath(os.path.join(appdata, "Sublime Text", "Packages"))
    sublime2 = os.path.abspath(os.path.join(appdata, "Sublime Text 3", "Packages"))
    neovim = os.path.abspath(os.path.join(local_appdata, "nvim-data", "site", "pack"))

    fake_paths = {sublime1, sublime2, neovim}

    def mock_isdir(p):
        return os.path.abspath(p) in fake_paths

    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    paths = get_editor_extensions_paths()

    assert sublime1 in paths
    assert sublime2 in paths
    assert neovim in paths

def test_get_editor_extensions_paths_win32_missing_env(monkeypatch):
    home = Path("C:/Users/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    def mock_isdir(p):
        return False

    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    paths = get_editor_extensions_paths()
    assert paths == []

def test_get_editor_extensions_paths_darwin(monkeypatch):
    home = Path("/Users/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(sys, "platform", "darwin")

    sublime1 = str(home / "Library" / "Application Support" / "Sublime Text" / "Packages")
    sublime2 = str(home / "Library" / "Application Support" / "Sublime Text 3" / "Packages")

    fake_paths = {os.path.abspath(sublime1), os.path.abspath(sublime2)}

    def mock_isdir(p):
        return os.path.abspath(p) in fake_paths

    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    paths = get_editor_extensions_paths()

    assert os.path.abspath(sublime1) in paths
    assert os.path.abspath(sublime2) in paths

def test_scan_editor_extensions_click(monkeypatch):
    target_paths = []
    def mock_set_scan_target(paths):
        nonlocal target_paths
        target_paths = paths

    def mock_button_click():
        pass

    monkeypatch.setattr("gptscan._set_scan_target", mock_set_scan_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)
    monkeypatch.setattr("gptscan.get_editor_extensions_paths", lambda: ["/mock/vscode/extensions"])

    scan_editor_extensions_click()
    assert target_paths == ["/mock/vscode/extensions"]

def test_scan_editor_extensions_click_no_paths(monkeypatch):
    monkeypatch.setattr("gptscan.get_editor_extensions_paths", lambda: [])

    message_box_shown = False
    def mock_showinfo(title, message):
        nonlocal message_box_shown
        message_box_shown = True
        assert "No editor extension folders" in message

    import gptscan
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_showinfo)

    scan_editor_extensions_click()
    assert message_box_shown
