import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path
import gptscan

def test_get_ruby_gems_paths(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def mock_check_output(cmd, **kwargs):
        if cmd == ['gem', 'env', 'home']:
            return "/home/user/.gems\n"
        raise subprocess.CalledProcessError(1, cmd)

    def mock_isdir(path):
        return path in [
            "/home/user/.gems",
            "/usr/local/lib/ruby/gems",
            "/usr/lib/ruby/gems",
            "/env/gems"
        ]

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir), \
         patch.dict(os.environ, {"GEM_HOME": "/env/gems"}):

        paths = gptscan.get_ruby_gems_paths()
        assert "/env/gems" in paths
        assert "/home/user/.gems" in paths
        assert "/usr/local/lib/ruby/gems" in paths
        assert "/usr/lib/ruby/gems" in paths

def test_get_php_packages_paths(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    mock_home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    def mock_check_output(cmd, **kwargs):
        if cmd == ['composer', 'global', 'config', 'vendor-dir', '--absolute']:
            return "/home/user/.config/composer/vendor\n"
        raise subprocess.CalledProcessError(1, cmd)

    def mock_isdir(path):
        return True

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir):

        paths = gptscan.get_php_packages_paths()
        assert "/home/user/.config/composer/vendor" in paths
        assert str(mock_home / ".composer" / "vendor") in paths

def test_get_rust_packages_paths(monkeypatch):
    mock_home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    def mock_isdir(path):
        return "registry" in path or "checkouts" in path

    with patch.dict(os.environ, {"CARGO_HOME": "/env/cargo"}), \
         patch("os.path.isdir", side_effect=mock_isdir):

        paths = gptscan.get_rust_packages_paths()
        assert os.path.join("/env/cargo", "registry", "src") in paths
        assert os.path.join("/env/cargo", "git", "checkouts") in paths
        assert str(mock_home / ".cargo" / "registry" / "src") in paths
        assert str(mock_home / ".cargo" / "git" / "checkouts") in paths

def test_get_go_packages_paths(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    mock_home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    def mock_check_output(cmd, **kwargs):
        if cmd == ['go', 'env', 'GOPATH']:
            return "/go/path1:/go/path2\n"
        raise subprocess.CalledProcessError(1, cmd)

    def mock_isdir(path):
        return True

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir), \
         patch.dict(os.environ, {"GOPATH": "/env/go"}):

        paths = gptscan.get_go_packages_paths()
        assert os.path.join("/env/go", "pkg", "mod") in paths
        assert os.path.join("/go/path1", "pkg", "mod") in paths
        assert os.path.join("/go/path2", "src") in paths
        assert str(mock_home / "go" / "pkg" / "mod") in paths

def test_get_documents_paths_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    mock_home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    def mock_exists(self):
        return "Documents" in str(self)

    def mock_isdir(path):
        return "Documents" in str(path)

    with patch("pathlib.Path.exists", side_effect=mock_exists, autospec=True), \
         patch("os.path.isdir", side_effect=mock_isdir):
        paths = gptscan.get_documents_paths()
        assert any("Documents" in p for p in paths)

def test_get_documents_paths_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    # We need to mock abspath and isdir to handle Windows-style paths on Linux test runner
    monkeypatch.setattr(os.path, "abspath", lambda x: x)
    monkeypatch.setattr(os.path, "isdir", lambda x: True)

    mock_home = MagicMock()
    mock_docs = MagicMock()
    mock_docs.exists.return_value = True
    mock_docs.__str__.return_value = "C:\\Users\\user\\Documents"
    mock_home.__truediv__.return_value = mock_docs

    monkeypatch.setattr("pathlib.Path.home", lambda: mock_home)

    # Mock winreg
    mock_winreg = MagicMock()
    mock_key = MagicMock()
    mock_winreg.OpenKey.return_value = mock_key
    mock_winreg.QueryValueEx.return_value = ("D:\\Custom\\Documents", None)
    monkeypatch.setitem(sys.modules, "winreg", mock_winreg)

    paths = gptscan.get_documents_paths()
    assert "C:\\Users\\user\\Documents" in paths
    assert "D:\\Custom\\Documents" in paths
