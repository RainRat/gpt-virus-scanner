import os
import sys
import pytest
import subprocess
from unittest.mock import MagicMock
import tkinter as tk
import gptscan
from gptscan import parse_report_content, Config, _get_initial_dir

def test_parse_report_content_undecipherable():
    invalid_content = "completely random non-json plain text with no structure"
    with pytest.raises(ValueError, match="Could not determine report format."):
        parse_report_content(invalid_content)

def test_config_save_cache_error(monkeypatch, capsys):
    mock_open = MagicMock(side_effect=OSError("Disk full"))
    monkeypatch.setattr("builtins.open", mock_open)

    Config.save_cache()

    captured = capsys.readouterr()
    assert "Warning: Could not save AI cache:" in captured.err
    assert "Disk full" in captured.err

def test_config_load_cache_error(monkeypatch, capsys):
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    mock_open = MagicMock(side_effect=OSError("Permission denied"))
    monkeypatch.setattr("builtins.open", mock_open)

    Config.load_cache()

    captured = capsys.readouterr()
    assert "Warning: Could not load AI cache:" in captured.err
    assert "Permission denied" in captured.err

def test_get_initial_dir_shlex_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = " # comment only "
    monkeypatch.setattr(gptscan, "textbox", mock_textbox)

    assert _get_initial_dir() is None

def test_is_supported_file_os_error(tmp_path):
    non_file_path = tmp_path / "test_directory"
    non_file_path.mkdir()

    assert Config.is_supported_file(non_file_path) is False

def test_get_system_path_directories_empty_path(monkeypatch):
    monkeypatch.delenv("PATH", raising=False)
    assert gptscan.get_system_path_directories() == []

    monkeypatch.setenv("PATH", "")
    assert gptscan.get_system_path_directories() == []

def test_get_shell_profile_paths_windows_exception(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    def mock_check_output(*args, **kwargs):
        raise subprocess.SubprocessError("Powershell failed")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr(os.path, "exists", lambda x: False)

    paths = gptscan.get_shell_profile_paths()
    assert isinstance(paths, list)

def test_get_php_packages_paths_composer_exception(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(127, "composer")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr(os.path, "isdir", lambda x: False)

    paths = gptscan.get_php_packages_paths()
    assert paths == []

def test_get_go_packages_paths_go_exception(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise OSError("go command not found")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr(os.path, "isdir", lambda x: False)

    paths = gptscan.get_go_packages_paths()
    assert paths == []

def test_update_progress_tcl_error(monkeypatch):
    mock_root = MagicMock()
    mock_progress_bar = MagicMock()

    def mock_getitem(key):
        if key == "maximum":
            raise tk.TclError("Tcl connection lost")
        return None
    mock_progress_bar.__getitem__.side_effect = mock_getitem

    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "progress_bar", mock_progress_bar)

    gptscan.update_progress(50)
    mock_root.update_idletasks.assert_called_once()

def test_update_progress_value_error(monkeypatch):
    mock_root = MagicMock()
    mock_progress_bar = MagicMock()
    mock_progress_bar.__getitem__.side_effect = lambda key: "not_a_float" if key == "maximum" else None

    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "progress_bar", mock_progress_bar)

    gptscan.update_progress(50)
    mock_root.update_idletasks.assert_called_once()
