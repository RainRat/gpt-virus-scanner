import pytest
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan

def test_get_shell_profile_paths_system_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    def mock_exists(self):
        valid = [
            "/etc/profile",
            "/etc/bash.bashrc",
            "/etc/environment",
            "/home/user/.bashrc"
        ]
        return str(self) in valid

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)

    paths = gptscan.get_shell_profile_paths()

    assert "/etc/profile" in paths
    assert "/etc/bash.bashrc" in paths
    assert "/etc/environment" in paths
    assert "/home/user/.bashrc" in paths

def test_get_shell_profile_paths_profile_d(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    def mock_exists(self):
        return str(self) == "/etc/profile.d"

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "is_dir", lambda self: str(self) == "/etc/profile.d")

    mock_sh = Path("/etc/profile.d/test.sh")
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [mock_sh] if pattern == "*.sh" else [])

    paths = gptscan.get_shell_profile_paths()

    assert "/etc/profile.d/test.sh" in paths

def test_get_shell_profile_paths_windows_robustness_list(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/user"))

    mock_output = json.dumps(["C:\\path1.ps1", "C:\\path2.ps1"])

    with patch("subprocess.check_output", return_value=mock_output), \
         patch("os.path.exists", return_value=True):
        paths = gptscan.get_shell_profile_paths()
        assert "C:\\path1.ps1" in paths
        assert "C:\\path2.ps1" in paths

def test_get_shell_profile_paths_windows_robustness_single(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/user"))

    mock_output = json.dumps("C:\\path1.ps1")

    with patch("subprocess.check_output", return_value=mock_output), \
         patch("os.path.exists", return_value=True):
        paths = gptscan.get_shell_profile_paths()
        assert "C:\\path1.ps1" in paths

def test_get_shell_profile_paths_windows_robustness_null(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/user"))

    mock_output = "null"

    with patch("subprocess.check_output", return_value=mock_output):
        paths = gptscan.get_shell_profile_paths()
        # Should not crash and just return user profiles (if any)
        assert isinstance(paths, list)

def test_get_shell_profile_paths_windows_robustness_invalid_type(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/user"))

    # PowerShell could theoretically return an object if ConvertTo-Json is used on a single object
    mock_output = json.dumps({"AllUsersAllHosts": "C:\\path1.ps1"})

    with patch("subprocess.check_output", return_value=mock_output):
        paths = gptscan.get_shell_profile_paths()
        # Should not crash
        assert isinstance(paths, list)
