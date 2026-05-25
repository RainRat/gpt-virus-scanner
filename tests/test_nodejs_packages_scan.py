import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
import pytest
import gptscan

def test_get_nodejs_package_paths_unix(monkeypatch):
    """Test Node.js package path detection on Unix-like systems."""
    monkeypatch.setattr(sys, "platform", "linux")

    # Mock npm root -g
    def mock_check_output(cmd, **kwargs):
        if cmd[:3] == ['npm', 'root', '-g']:
            return "/home/user/.nvm/versions/node/v20.0.0/lib/node_modules\n"
        raise subprocess.CalledProcessError(1, cmd)

    # Mock os.path.isdir to return True for our expected paths
    def mock_isdir(path):
        return path in [
            "/home/user/.nvm/versions/node/v20.0.0/lib/node_modules",
            "/usr/local/lib/node_modules",
            "/usr/lib/node_modules"
        ]

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir):
        paths = gptscan.get_nodejs_package_paths()

        assert "/home/user/.nvm/versions/node/v20.0.0/lib/node_modules" in paths
        assert "/usr/local/lib/node_modules" in paths
        assert "/usr/lib/node_modules" in paths

def test_get_nodejs_package_paths_windows(monkeypatch):
    """Test Node.js package path detection on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")

    # Mock npm root -g
    def mock_check_output(cmd, **kwargs):
        if cmd[:3] == ['npm', 'root', '-g']:
            return "C:\\Users\\User\\AppData\\Roaming\\npm\\node_modules\r\n"
        raise subprocess.CalledProcessError(1, cmd)

    # Mock environment variables
    monkeypatch.setenv("APPDATA", "C:\\Users\\User\\AppData\\Roaming")
    monkeypatch.setenv("ProgramFiles", "C:\\Program Files")

    # Mock os.path.isdir
    def mock_isdir(path):
            return True

    with patch("subprocess.check_output", side_effect=mock_check_output), \
         patch("os.path.isdir", side_effect=mock_isdir):
        # We need to mock abspath and join to be platform-agnostic for Windows paths on Linux
        monkeypatch.setattr(os.path, "abspath", lambda x: x)
        monkeypatch.setattr(os.path, "join", lambda *args: "\\".join(args))

        paths = gptscan.get_nodejs_package_paths()

        assert "C:\\Users\\User\\AppData\\Roaming\\npm\\node_modules" in paths
        assert "C:\\Program Files\\nodejs\\node_modules" in paths

def test_get_nodejs_package_paths_no_npm(monkeypatch):
    """Test behavior when npm is not found."""
    monkeypatch.setattr(sys, "platform", "linux")

    with patch("subprocess.check_output", side_effect=FileNotFoundError), \
         patch("os.path.isdir", return_value=True):
        paths = gptscan.get_nodejs_package_paths()
        # Should still contain common system paths
        assert "/usr/local/lib/node_modules" in paths
        assert "/usr/lib/node_modules" in paths

def test_system_audit_includes_nodejs(monkeypatch):
    """Verify that System Audit includes Node.js package paths."""
    with patch("gptscan.get_nodejs_package_paths", return_value=["/mock/node_modules"]), \
         patch("gptscan.get_shell_profile_paths", return_value=[]), \
         patch("gptscan.get_shell_history_paths", return_value=[]), \
         patch("gptscan.get_system_path_directories", return_value=[]), \
         patch("gptscan.get_ssh_config_paths", return_value=[]), \
         patch("gptscan.get_system_service_paths", return_value=[]), \
         patch("gptscan.get_git_hooks_paths", return_value=[]), \
         patch("gptscan.get_python_package_paths", return_value=[]):

        all_paths, _ = gptscan.get_system_audit_data()
        assert "/mock/node_modules" in all_paths
