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

def test_get_shell_profile_paths_posix_system_profiles(tmp_path, monkeypatch):
    fake_etc = tmp_path / "etc"
    fake_etc.mkdir()
    (fake_etc / "profile").write_text("# /etc/profile")
    (fake_etc / "bash.bashrc").write_text("# bash.bashrc")
    profile_d = fake_etc / "profile.d"
    profile_d.mkdir()
    (profile_d / "custom.sh").write_text("# custom script")
    (profile_d / "ignored.txt").write_text("not a sh file")

    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setattr("sys.platform", "linux")

    # Patch Path constructor in gptscan module to route /etc system paths to fake_etc
    orig_path_cls = Path

    def mock_path(*args, **kwargs):
        if args and isinstance(args[0], (str, Path)):
            p_str = str(args[0])
            if p_str == '/etc/profile':
                return fake_etc / "profile"
            if p_str == '/etc/bash.bashrc':
                return fake_etc / "bash.bashrc"
            if p_str == '/etc/environment':
                return fake_etc / "environment"
            if p_str == '/etc/profile.d':
                return profile_d
        return orig_path_cls(*args, **kwargs)

    mock_path.home = staticmethod(lambda: home)

    monkeypatch.setattr(gptscan, "Path", mock_path)

    paths = gptscan.get_shell_profile_paths()
    assert str(fake_etc / "profile") in paths
    assert str(fake_etc / "bash.bashrc") in paths
    assert str(profile_d / "custom.sh") in paths
    assert str(profile_d / "ignored.txt") not in paths

def test_get_shell_profile_paths_windows_error_handling(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    def mock_check_output_error(*args, **kwargs):
        raise subprocess.SubprocessError("PowerShell unavailable")

    monkeypatch.setattr("subprocess.check_output", mock_check_output_error)
    paths = gptscan.get_shell_profile_paths()
    assert isinstance(paths, list)

    def mock_check_output_bad_json(*args, **kwargs):
        return "invalid json {"

    monkeypatch.setattr("subprocess.check_output", mock_check_output_bad_json)
    paths = gptscan.get_shell_profile_paths()
    assert isinstance(paths, list)

def test_scan_shell_profiles_click_success(monkeypatch):
    called_target = []
    called_click = []

    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: ["/fake/.bashrc"])
    monkeypatch.setattr("gptscan._set_scan_target", lambda path: called_target.append(path))
    monkeypatch.setattr("gptscan.button_click", lambda: called_click.append(True))

    gptscan.scan_shell_profiles_click()

    assert called_target == [["/fake/.bashrc"]]
    assert called_click == [True]

def test_scan_shell_profiles_click_no_files(monkeypatch):
    called_info = []

    monkeypatch.setattr("gptscan.get_shell_profile_paths", lambda: [])
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda title, msg: called_info.append((title, msg)))

    gptscan.scan_shell_profiles_click()

    assert len(called_info) == 1
    assert called_info[0][0] == "Shell Profiles"
