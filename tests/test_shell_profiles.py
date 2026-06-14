import pytest
import os
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
        # Filter to home-relative paths for environment-agnostic testing
        home_paths = [p for p in paths if p.startswith(str(home))]
        assert str(bashrc) in home_paths
        assert str(profile) in home_paths
        assert len(home_paths) == 2

def test_get_shell_profile_paths_system_wide():
    """Verify discovery of system-wide profile files on POSIX."""
    # Mocking os.path.exists and Path.glob/exists for system paths
    with patch("sys.platform", "linux"), \
         patch("gptscan.Path") as mock_path:

        # Configure mock_path to handle /etc/profile.d
        mock_profile_d = MagicMock()
        mock_profile_d.exists.return_value = True
        mock_profile_d.is_dir.return_value = True
        mock_profile_d.glob.return_value = [Path("/etc/profile.d/test.sh")]

        def path_side_effect(p):
            if p == "/etc/profile.d":
                return mock_profile_d
            return Path(p)

        mock_path.side_effect = path_side_effect
        mock_path.home.return_value = Path("/nonexistent/home")

        with patch("os.path.exists") as mock_exists:
            # We want to match exactly what get_shell_profile_paths checks
            mock_exists.side_effect = lambda p: p in ["/etc/profile", "/nonexistent/home/.bashrc"]

            paths = gptscan.get_shell_profile_paths()

            assert "/etc/profile" in paths
            assert "/etc/profile.d/test.sh" in paths

@patch("sys.platform", "win32")
@patch("subprocess.check_output")
@patch("os.path.exists")
def test_get_shell_profile_paths_windows(mock_exists, mock_check_output, tmp_path):
    mock_check_output.return_value = '"C:\\\\Users\\\\user\\\\Documents\\\\WindowsPowerShell\\\\Microsoft.PowerShell_profile.ps1"'
    mock_exists.return_value = True

    with patch("pathlib.Path.home", return_value=tmp_path):
        paths = gptscan.get_shell_profile_paths()
        assert "C:\\Users\\user\\Documents\\WindowsPowerShell\\Microsoft.PowerShell_profile.ps1" in paths

def test_scan_shell_profiles_click_success(mocker):
    mock_get_paths = mocker.patch("gptscan.get_shell_profile_paths", return_value=["/fake/.bashrc"])
    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")

    gptscan.scan_shell_profiles_click()

    mock_get_paths.assert_called_once()
    mock_set_target.assert_called_once_with(["/fake/.bashrc"])
    mock_button_click.assert_called_once()

def test_scan_shell_profiles_click_no_files(mocker):
    mock_get_paths = mocker.patch("gptscan.get_shell_profile_paths", return_value=[])
    mock_messagebox = mocker.patch("tkinter.messagebox.showinfo")

    gptscan.scan_shell_profiles_click()

    mock_messagebox.assert_called_once()
