import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan

def test_get_shell_profile_paths_posix_system_wide(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    # Mock Path.home()
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", lambda: home)

    # Files that exist
    existing_files = {
        str(home / ".bashrc"),
        "/etc/profile",
        "/etc/environment",
        "/etc/profile.d/test.sh",
        "/etc/profile.d/other.sh"
    }

    # Directories that exist
    existing_dirs = {
        "/etc/profile.d"
    }

    def mock_exists(self):
        return str(self) in existing_files or str(self) in existing_dirs

    def mock_is_dir(self):
        return str(self) in existing_dirs

    def mock_glob(self, pattern):
        if str(self) == "/etc/profile.d" and pattern == "*.sh":
            return [Path("/etc/profile.d/test.sh"), Path("/etc/profile.d/other.sh")]
        return []

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "is_dir", mock_is_dir)
    monkeypatch.setattr(Path, "glob", mock_glob)

    paths = gptscan.get_shell_profile_paths()

    assert "/etc/profile" in paths
    assert "/etc/environment" in paths
    assert "/etc/profile.d/test.sh" in paths
    assert "/etc/profile.d/other.sh" in paths
    assert str(home / ".bashrc") in paths
    # /etc/bash.bashrc was not in existing_files
    assert "/etc/bash.bashrc" not in paths
