import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan

def test_get_documents_paths(monkeypatch):
    home = Path.home()
    docs = home / "Documents"

    # Mock exists to return True for Documents
    real_exists = Path.exists
    def mock_exists(self):
        if str(self) == str(docs):
            return True
        return False

    monkeypatch.setattr(Path, "exists", mock_exists)

    paths = gptscan.get_documents_paths()
    assert str(docs) in paths

def test_get_ruby_gems_paths(monkeypatch):
    monkeypatch.setenv("GEM_HOME", "/mock/gem/home")

    # Mock subprocess to return a path string
    def mock_check_output(cmd, **kwargs):
        if cmd == ['gem', 'env', 'home']:
            return "/mock/gem/env/home\n"
        return ""

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # Mock os.path.isdir for the mocked paths
    def mock_isdir(path):
        # Handle both string and bytes just in case, though it should be string now
        p_str = path.decode() if isinstance(path, bytes) else str(path)
        return p_str in ["/mock/gem/home", "/mock/gem/env/home"]

    monkeypatch.setattr("os.path.isdir", mock_isdir)

    paths = gptscan.get_ruby_gems_paths()
    assert "/mock/gem/home" in paths
    assert "/mock/gem/env/home" in paths

def test_get_php_packages_paths(monkeypatch):
    # Mock subprocess to return a path string
    def mock_check_output(cmd, **kwargs):
        if 'composer' in cmd:
            return "/mock/composer/vendor\n"
        return ""

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # Mock fallback paths
    home = Path.home()
    p1 = str(home / ".composer" / "vendor")
    p2 = str(home / ".config" / "composer" / "vendor")

    def mock_isdir(path):
        p_str = path.decode() if isinstance(path, bytes) else str(path)
        return p_str in ["/mock/composer/vendor", p1, p2]

    monkeypatch.setattr("os.path.isdir", mock_isdir)

    paths = gptscan.get_php_packages_paths()
    assert "/mock/composer/vendor" in paths
    assert p1 in paths

def test_get_rust_packages_paths(monkeypatch):
    monkeypatch.setenv("CARGO_HOME", "/mock/cargo")

    home = Path.home()
    p1 = str(home / ".cargo" / "registry" / "src")
    p2 = "/mock/cargo/registry/src"

    def mock_isdir(path):
        p_str = path.decode() if isinstance(path, bytes) else str(path)
        return p_str in [p1, p2]

    monkeypatch.setattr("os.path.isdir", mock_isdir)

    paths = gptscan.get_rust_packages_paths()
    assert p1 in paths
    assert p2 in paths

def test_get_go_packages_paths(monkeypatch):
    monkeypatch.setenv("GOPATH", "/mock/go")

    home = Path.home()
    p1 = str(home / "go" / "pkg" / "mod")
    p2 = "/mock/go/pkg/mod"

    # Also mock subprocess for go env GOPATH
    def mock_check_output(cmd, **kwargs):
        if 'go' in cmd:
            return "/mock/go/env\n"
        return ""
    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    p3 = "/mock/go/env/pkg/mod"
    def mock_isdir(path):
        p_str = path.decode() if isinstance(path, bytes) else str(path)
        return p_str in [p1, p2, p3]
    monkeypatch.setattr("os.path.isdir", mock_isdir)

    paths = gptscan.get_go_packages_paths()
    assert p1 in paths
    assert p2 in paths
    assert p3 in paths

def test_system_audit_data_includes_new_paths(monkeypatch):
    # Mock all discovery functions to return unique markers
    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: ["/m_profile"])
    monkeypatch.setattr("gptscan.get_shell_history_paths", lambda: ["/m_history"])
    monkeypatch.setattr("gptscan.get_system_path_directories", lambda: ["/m_syspath"])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", lambda: ["/m_ssh"])
    monkeypatch.setattr("gptscan.get_system_service_paths", lambda: ["/m_service"])
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda: ["/m_hooks"])
    monkeypatch.setattr("gptscan.get_python_package_paths", lambda: ["/m_python"])
    monkeypatch.setattr("gptscan.get_nodejs_package_paths", lambda: ["/m_node"])
    monkeypatch.setattr("gptscan.get_browser_extensions_paths", lambda: ["/m_browser"])
    monkeypatch.setattr("gptscan.get_editor_extensions_paths", lambda: ["/m_editor"])
    monkeypatch.setattr("gptscan.get_ruby_gems_paths", lambda: ["/m_ruby"])
    monkeypatch.setattr("gptscan.get_php_packages_paths", lambda: ["/m_php"])
    monkeypatch.setattr("gptscan.get_rust_packages_paths", lambda: ["/m_rust"])
    monkeypatch.setattr("gptscan.get_go_packages_paths", lambda: ["/m_go"])
    monkeypatch.setattr("gptscan.get_documents_paths", lambda: ["/m_docs"])
    monkeypatch.setattr("gptscan.get_downloads_paths", lambda: ["/m_downloads"])
    monkeypatch.setattr("gptscan.get_desktop_paths", lambda: ["/m_desktop"])
    monkeypatch.setattr("gptscan.get_temp_paths", lambda: ["/m_temp"])

    paths, _ = gptscan.get_system_audit_data()

    assert "/m_ruby" in paths
    assert "/m_php" in paths
    assert "/m_rust" in paths
    assert "/m_go" in paths
    assert "/m_docs" in paths
