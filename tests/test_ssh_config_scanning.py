import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan

def test_get_ssh_config_paths_linux(monkeypatch, tmp_path):
    """Test SSH config path detection on Linux."""
    monkeypatch.setattr(sys, "platform", "linux")

    home = tmp_path / "home" / "user"
    home.mkdir(parents=True)
    user_ssh = home / ".ssh"
    user_ssh.mkdir()
    (user_ssh / "config").write_text("Host *")
    (user_ssh / "authorized_keys").write_text("ssh-rsa AAA...")

    etc_ssh = tmp_path / "etc" / "ssh"
    etc_ssh.mkdir(parents=True)
    (etc_ssh / "sshd_config").write_text("Port 22")
    (etc_ssh / "ssh_config").write_text("Host *")

    monkeypatch.setattr(Path, "home", lambda: home)

    # Mock Path.exists to check our tmp_path structure
    original_exists = Path.exists
    def mock_exists(self):
        # Resolve path relative to tmp_path if it's absolute from root
        path_str = str(self)
        if path_str.startswith("/etc/ssh"):
            rel_path = path_str[1:] # etc/ssh/...
            return (tmp_path / rel_path).exists()
        return original_exists(self)

    with patch.object(Path, "exists", autospec=True) as mock_path_exists:
        mock_path_exists.side_effect = mock_exists
        paths = gptscan.get_ssh_config_paths()

    # On linux, it should find user and system files
    # Note: in our mock, we only handled /etc/ssh for system files
    # User files are found via Path.home() which we mocked to tmp_path/home/user

    expected_user_config = str(user_ssh / "config")
    expected_user_keys = str(user_ssh / "authorized_keys")
    expected_system_sshd = "/etc/ssh/sshd_config"
    expected_system_ssh = "/etc/ssh/ssh_config"

    assert expected_user_config in paths
    assert expected_user_keys in paths
    assert expected_system_sshd in paths
    assert expected_system_ssh in paths

def test_get_ssh_config_paths_windows(monkeypatch, tmp_path):
    """Test SSH config path detection on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")

    home = tmp_path / "Users" / "user"
    home.mkdir(parents=True)
    user_ssh = home / ".ssh"
    user_ssh.mkdir()
    (user_ssh / "config").write_text("Host *")

    program_data = tmp_path / "ProgramData"
    win_system_ssh = program_data / "ssh"
    win_system_ssh.mkdir(parents=True)
    (win_system_ssh / "sshd_config").write_text("Port 22")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("ProgramData", str(program_data))

    # We need to mock os.path.exists as well because Path.exists calls it sometimes
    # and gptscan might use it directly if refactored, but currently it uses Path.exists

    # For Windows test on Linux runner, we must be careful with Path separators
    # gptscan uses Path("/etc/ssh") which is absolute on Linux

    def mock_exists(self):
        p_str = str(self)
        if p_str == "/etc/ssh":
            return False # Simulate no Linux system path on "Windows"
        return original_exists(self)

    original_exists = Path.exists
    with patch.object(Path, "exists", autospec=True) as mock_path_exists:
        mock_path_exists.side_effect = mock_exists
        paths = gptscan.get_ssh_config_paths()

    assert str(user_ssh / "config") in paths
    assert str(win_system_ssh / "sshd_config") in paths
    assert "/etc/ssh/sshd_config" not in paths

def test_get_ssh_config_paths_missing(monkeypatch, tmp_path):
    """Test behavior when no SSH files exist."""
    home = tmp_path / "empty_home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    # Ensure /etc/ssh doesn't exist (it shouldn't in tmp_path)
    # Mock Path.exists to return False for everything except the home dir
    def mock_exists(self):
        if str(self) == str(home):
            return True
        return False

    with patch.object(Path, "exists", autospec=True) as mock_path_exists:
        mock_path_exists.side_effect = mock_exists
        paths = gptscan.get_ssh_config_paths()

    assert paths == []
