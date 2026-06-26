import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import gptscan

def test_get_ruby_gems_paths(monkeypatch, tmp_path):
    gem_home = str(tmp_path / "gem_home")
    gem_env_home = str(tmp_path / "gem_env_home")

    monkeypatch.setenv("GEM_HOME", gem_home)
    Path(gem_home).mkdir()
    Path(gem_env_home).mkdir()

    def mock_check_output(cmd, **kwargs):
        if cmd == ['gem', 'env', 'home']:
            return gem_env_home + "\n"
        raise subprocess.CalledProcessError(1, cmd)

    def mock_isdir(path):
        return path in [gem_home, gem_env_home, "/usr/local/lib/ruby/gems"]

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir), \
         patch("sys.platform", "linux"):

        # We need to reach into the function to make sure it includes the fallbacks
        with patch("gptscan._normalize_and_filter_dirs") as mock_norm:
            mock_norm.side_effect = lambda paths: [p for p in paths if mock_isdir(p)]
            paths = gptscan.get_ruby_gems_paths()

    assert gem_home in paths
    assert gem_env_home in paths
    assert "/usr/local/lib/ruby/gems" in paths

def test_get_php_packages_paths(monkeypatch, tmp_path):
    composer_vendor = str(tmp_path / "composer_vendor")
    Path(composer_vendor).mkdir()
    home_composer = str(tmp_path / ".composer" / "vendor")
    Path(home_composer).mkdir(parents=True)

    def mock_check_output(cmd, **kwargs):
        if 'composer' in cmd:
            return composer_vendor + "\n"
        raise subprocess.CalledProcessError(1, cmd)

    def mock_isdir(path):
        return path in [composer_vendor, home_composer]

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir), \
         patch("pathlib.Path.home", return_value=tmp_path):

        with patch("gptscan._normalize_and_filter_dirs") as mock_norm:
            mock_norm.side_effect = lambda paths: [p for p in paths if mock_isdir(p)]
            paths = gptscan.get_php_packages_paths()

    assert composer_vendor in paths
    assert home_composer in paths

def test_get_rust_packages_paths(monkeypatch, tmp_path):
    cargo_home = str(tmp_path / "cargo")
    monkeypatch.setenv("CARGO_HOME", cargo_home)
    Path(cargo_home, "registry", "src").mkdir(parents=True)
    Path(cargo_home, "git", "checkouts").mkdir(parents=True)

    home_cargo = tmp_path / "home_cargo"
    Path(home_cargo, ".cargo", "registry", "src").mkdir(parents=True)

    def mock_isdir(path):
        return True

    with patch("os.path.isdir", return_value=True), \
         patch("pathlib.Path.home", return_value=home_cargo):
        paths = gptscan.get_rust_packages_paths()

    assert str(Path(cargo_home, "registry", "src")) in paths
    assert str(Path(home_cargo, ".cargo", "registry", "src")) in paths

def test_get_go_packages_paths(monkeypatch, tmp_path):
    go_env = str(tmp_path / "go_env")
    monkeypatch.setenv("GOPATH", go_env)
    Path(go_env, "pkg", "mod").mkdir(parents=True)

    go_cmd = str(tmp_path / "go_cmd")
    Path(go_cmd, "pkg", "mod").mkdir(parents=True)

    def mock_check_output(cmd, **kwargs):
        if 'go' in cmd:
            return go_cmd + "\n"
        raise subprocess.CalledProcessError(1, cmd)

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", return_value=True), \
         patch("pathlib.Path.home", return_value=tmp_path):
        paths = gptscan.get_go_packages_paths()

    assert str(Path(go_env, "pkg", "mod")) in paths
    assert str(Path(go_cmd, "pkg", "mod")) in paths
    assert str(tmp_path / "go" / "pkg" / "mod") in paths

def test_get_documents_paths_linux(monkeypatch, tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()

    with patch("sys.platform", "linux"), \
         patch("pathlib.Path.home", return_value=tmp_path), \
         patch("os.path.isdir", return_value=True):
        paths = gptscan.get_documents_paths()

    assert str(docs) in paths

def test_get_documents_paths_windows(monkeypatch, tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()
    reg_docs = str(tmp_path / "RegDocuments")
    Path(reg_docs).mkdir()

    monkeypatch.setattr(sys, "platform", "win32")

    mock_winreg = MagicMock()
    monkeypatch.setitem(sys.modules, "winreg", mock_winreg)
    mock_winreg.OpenKey.return_value = "dummy_key"
    mock_winreg.QueryValueEx.return_value = (reg_docs, "REG_SZ")

    def mock_isdir(path):
        return True

    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("os.path.isdir", side_effect=mock_isdir):
        paths = gptscan.get_documents_paths()

    assert str(docs) in paths
    assert reg_docs in paths
