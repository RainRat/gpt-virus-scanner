import pytest
import os
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import gptscan

def test_get_shell_profile_paths_linux(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    bashrc = home / ".bashrc"
    bashrc.write_text("alias ls='ls --color=auto'")
    profile = home / ".profile"
    profile.write_text("export PATH=$PATH:~/bin")

    with patch("pathlib.Path.home", return_value=home):
        paths = gptscan.get_shell_profile_paths()
        home_paths = [p for p in paths if p.startswith(str(home))]
        assert str(bashrc) in home_paths
        assert str(profile) in home_paths
        assert len(home_paths) == 2

@patch("sys.platform", "win32")
@patch("subprocess.check_output")
@patch("os.path.exists")
def test_get_shell_profile_paths_windows(mock_exists, mock_check_output, tmp_path):
    mock_check_output.return_value = '"C:\\\\Users\\\\user\\\\Documents\\\\WindowsPowerShell\\\\Microsoft.PowerShell_profile.ps1"'
    mock_exists.return_value = True

    with patch("pathlib.Path.home", return_value=tmp_path):
        paths = gptscan.get_shell_profile_paths()
        assert "C:\\Users\\user\\Documents\\WindowsPowerShell\\Microsoft.PowerShell_profile.ps1" in paths

def test_get_shell_profile_paths_posix_system(monkeypatch):
    fake_home = Path("/fake/home/user")

    def mock_exists(path_obj):
        p_str = str(path_obj)
        if p_str in ("/etc/profile", "/etc/bash.bashrc", "/etc/environment"):
            return True
        if p_str == "/etc/profile.d":
            return True
        if p_str.startswith("/fake/home"):
            return False
        return False

    def mock_is_dir(path_obj):
        p_str = str(path_obj)
        return p_str == "/etc/profile.d"

    def mock_glob(path_obj, pattern):
        if pattern == "*.sh":
            return [Path("/etc/profile.d/custom.sh")]
        return []

    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "is_dir", mock_is_dir)
    monkeypatch.setattr(Path, "glob", mock_glob)

    paths = gptscan.get_shell_profile_paths()
    assert "/etc/profile" in paths
    assert "/etc/bash.bashrc" in paths
    assert "/etc/environment" in paths
    assert "/etc/profile.d/custom.sh" in paths

@patch("sys.platform", "win32")
def test_get_shell_profile_paths_windows_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # Test SubprocessError
    with patch("subprocess.check_output", side_effect=subprocess.SubprocessError("powershell error")):
        paths = gptscan.get_shell_profile_paths()
        assert paths == []

    # Test JSONDecodeError
    with patch("subprocess.check_output", return_value="invalid json"):
        paths = gptscan.get_shell_profile_paths()
        assert paths == []

def test_scan_shell_profiles_click_success(monkeypatch):
    mock_get_paths = MagicMock(return_value=["/fake/.bashrc"])
    mock_set_target = MagicMock()
    mock_button_click = MagicMock()

    monkeypatch.setattr("gptscan.get_shell_profile_paths", mock_get_paths)
    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_shell_profiles_click()

    mock_get_paths.assert_called_once()
    mock_set_target.assert_called_once_with(["/fake/.bashrc"])
    mock_button_click.assert_called_once()

def test_scan_shell_profiles_click_no_files(monkeypatch):
    mock_get_paths = MagicMock(return_value=[])
    mock_messagebox = MagicMock()

    monkeypatch.setattr("gptscan.get_shell_profile_paths", mock_get_paths)
    monkeypatch.setattr("tkinter.messagebox.showinfo", mock_messagebox)

    gptscan.scan_shell_profiles_click()

    mock_messagebox.assert_called_once()
