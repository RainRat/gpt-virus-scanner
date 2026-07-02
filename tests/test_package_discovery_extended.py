import os
import sys
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan
from gptscan import (
    get_ruby_gems_paths,
    get_php_packages_paths,
    get_rust_packages_paths,
    get_go_packages_paths,
    get_documents_paths
)

def _norm(p):
    return os.path.abspath(p).replace("\\", "/").replace("//", "/")

def test_get_ruby_gems_paths(monkeypatch):
    """Verify get_ruby_gems_paths logic including env vars, subprocess, and fallbacks."""
    # Mock os.path.isdir to return True for our fake paths
    monkeypatch.setattr(os.path, "isdir", lambda x: True)

    # 1. Test GEM_HOME
    monkeypatch.setenv("GEM_HOME", "/fake/gem_home")

    # 2. Test gem subprocess
    def mock_check_output(args, **kwargs):
        if args == ['gem', 'env', 'home']:
            return "/fake/gem_subprocess_home\n"
        return ""
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # 3. Test fallbacks (non-win32)
    monkeypatch.setattr(sys, "platform", "linux")

    paths = [_norm(p) for p in get_ruby_gems_paths()]

    assert _norm("/fake/gem_home") in paths
    assert _norm("/fake/gem_subprocess_home") in paths
    assert _norm("/usr/local/lib/ruby/gems") in paths
    assert _norm("/usr/lib/ruby/gems") in paths

def test_get_ruby_gems_paths_no_gem(monkeypatch):
    """Verify get_ruby_gems_paths handles gem command failure."""
    monkeypatch.setattr(os.path, "isdir", lambda x: True)
    monkeypatch.delenv("GEM_HOME", raising=False)

    def mock_check_output(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    monkeypatch.setattr(sys, "platform", "linux")

    paths = [_norm(p) for p in get_ruby_gems_paths()]
    # Should still have fallbacks
    assert _norm("/usr/local/lib/ruby/gems") in paths
    assert _norm("/usr/lib/ruby/gems") in paths
    # Should not have subprocess path
    assert _norm("/fake/gem_subprocess_home") not in paths

def test_get_php_packages_paths_linux(monkeypatch, tmp_path):
    """Verify get_php_packages_paths on Linux."""
    monkeypatch.setattr(os.path, "isdir", lambda x: True)
    monkeypatch.setattr(sys, "platform", "linux")

    # Mock home
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # Mock composer subprocess
    def mock_check_output(args, **kwargs):
        if 'composer' in args:
            return "/fake/composer/vendor\n"
        return ""
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    paths = [_norm(p) for p in get_php_packages_paths()]

    assert _norm("/fake/composer/vendor") in paths
    assert _norm(str(fake_home / ".composer" / "vendor")) in paths
    assert _norm(str(fake_home / ".config" / "composer" / "vendor")) in paths

def test_get_php_packages_paths_windows(monkeypatch, tmp_path):
    """Verify get_php_packages_paths on Windows."""
    monkeypatch.setattr(os.path, "isdir", lambda x: True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:\\Users\\User\\AppData\\Roaming")

    # Mock composer subprocess returning Windows path
    def mock_check_output(args, **kwargs):
        return "C:\\Composer\\vendor\n"
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    paths = [_norm(p) for p in get_php_packages_paths()]

    assert _norm("C:\\Composer\\vendor") in paths
    assert _norm("C:\\Users\\User\\AppData\\Roaming\\Composer\\vendor") in paths

def test_get_rust_packages_paths(monkeypatch, tmp_path):
    """Verify get_rust_packages_paths logic."""
    monkeypatch.setattr(os.path, "isdir", lambda x: True)

    # 1. Test CARGO_HOME
    monkeypatch.setenv("CARGO_HOME", "/fake/cargo_home")

    # 2. Test default home
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    paths = [_norm(p) for p in get_rust_packages_paths()]

    assert _norm("/fake/cargo_home/registry/src") in paths
    assert _norm("/fake/cargo_home/git/checkouts") in paths
    assert _norm(str(fake_home / ".cargo" / "registry" / "src")) in paths
    assert _norm(str(fake_home / ".cargo" / "git" / "checkouts")) in paths

def test_get_go_packages_paths(monkeypatch, tmp_path):
    """Verify get_go_packages_paths logic."""
    monkeypatch.setattr(os.path, "isdir", lambda x: True)

    # 1. Test GOPATH env
    monkeypatch.setenv("GOPATH", "/fake/gopath1" + os.pathsep + "/fake/gopath2")

    # 2. Test go env GOPATH subprocess
    def mock_check_output(args, **kwargs):
        if 'go' in args and 'GOPATH' in args:
            return "/fake/go_subprocess_path\n"
        return ""
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # 3. Test default location
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    paths = [_norm(p) for p in get_go_packages_paths()]

    assert _norm(os.path.join("/fake/gopath1", "pkg", "mod")) in paths
    assert _norm(os.path.join("/fake/gopath1", "src")) in paths
    assert _norm(os.path.join("/fake/gopath2", "pkg", "mod")) in paths
    assert _norm(os.path.join("/fake/gopath2", "src")) in paths
    assert _norm(os.path.join("/fake/go_subprocess_path", "pkg", "mod")) in paths
    assert _norm(os.path.join("/fake/go_subprocess_path", "src")) in paths
    assert _norm(str(fake_home / "go" / "pkg" / "mod")) in paths
    assert _norm(str(fake_home / "go" / "src")) in paths

def test_get_documents_paths_posix(monkeypatch, tmp_path):
    """Verify get_documents_paths on POSIX."""
    monkeypatch.setattr(sys, "platform", "linux")
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    docs = fake_home / "Documents"
    docs.mkdir()

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # Patch os.path.isdir to return True for our fake docs
    monkeypatch.setattr(os.path, "isdir", lambda x: _norm(x) == _norm(str(docs)))

    paths = [_norm(p) for p in get_documents_paths()]
    assert _norm(str(docs)) in paths

def test_get_documents_paths_windows(monkeypatch, tmp_path):
    """Verify get_documents_paths on Windows with winreg mock."""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # Mock winreg
    mock_winreg = MagicMock()
    monkeypatch.setitem(sys.modules, "winreg", mock_winreg)

    mock_key = MagicMock()
    mock_winreg.OpenKey.return_value = mock_key
    mock_winreg.QueryValueEx.return_value = ("C:\\Users\\User\\Documents", None)
    mock_winreg.HKEY_CURRENT_USER = "HKCU"

    # Mock os.path.isdir to return True for the winreg path
    def mock_isdir(p):
        return _norm(p) == _norm("C:\\Users\\User\\Documents")
    monkeypatch.setattr(os.path, "isdir", mock_isdir)

    # Mock Path.exists to return False for default home/Documents to focus on winreg
    monkeypatch.setattr(Path, "exists", lambda self: False)

    paths = [_norm(p) for p in get_documents_paths()]
    assert _norm("C:\\Users\\User\\Documents") in paths
    mock_winreg.OpenKey.assert_called_once()
