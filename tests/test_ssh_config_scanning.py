import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import gptscan

def test_get_ssh_config_paths_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    # Mock Path.exists to control which files are found
    def mock_exists(self):
        valid_paths = [
            "/home/user/.ssh",
            "/home/user/.ssh/config",
            "/home/user/.ssh/authorized_keys",
            "/etc/ssh",
            "/etc/ssh/sshd_config",
            "/etc/ssh/ssh_config"
        ]
        return str(self).replace("\\", "/") in valid_paths

    monkeypatch.setattr(Path, "exists", mock_exists)

    paths = [p.replace("\\", "/") for p in gptscan.get_ssh_config_paths()]

    assert "/home/user/.ssh/config" in paths
    assert "/home/user/.ssh/authorized_keys" in paths
    assert "/etc/ssh/sshd_config" in paths
    assert "/etc/ssh/ssh_config" in paths
    assert len(paths) == 4

def test_get_ssh_config_paths_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    home = Path("C:/Users/user")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("ProgramData", "C:/ProgramData")

    def mock_exists(self):
        valid_paths = [
            "C:/Users/user/.ssh",
            "C:/Users/user/.ssh/config",
            "C:/ProgramData/ssh",
            "C:/ProgramData/ssh/sshd_config"
        ]
        return str(self).replace("\\", "/") in valid_paths

    monkeypatch.setattr(Path, "exists", mock_exists)

    paths = [p.replace("\\", "/") for p in gptscan.get_ssh_config_paths()]

    assert "C:/Users/user/.ssh/config" in paths
    assert "C:/ProgramData/ssh/sshd_config" in paths
    assert "C:/Users/user/.ssh/authorized_keys" not in paths
    assert "/etc/ssh/sshd_config" not in paths
    assert len(paths) == 2

def test_get_ssh_config_paths_empty(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    paths = gptscan.get_ssh_config_paths()
    assert paths == []
