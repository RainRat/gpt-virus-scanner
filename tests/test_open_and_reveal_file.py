import os
import sys
import subprocess
from unittest.mock import MagicMock
import pytest
import gptscan


def test_open_file_oserror(monkeypatch):
    monkeypatch.setattr(gptscan, "_resolve_file_paths", lambda evt: ["/path/to/file.py"])
    monkeypatch.setattr(sys, "platform", "linux")

    def mock_run(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(subprocess, "run", mock_run)

    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    gptscan.open_file()

    mock_showerror.assert_called_once_with("Error", "Could not open file '/path/to/file.py': Permission denied")


def test_open_file_subprocess_error(monkeypatch):
    monkeypatch.setattr(gptscan, "_resolve_file_paths", lambda evt: ["/path/to/file.py"])
    monkeypatch.setattr(sys, "platform", "darwin")

    def mock_run(*args, **kwargs):
        raise subprocess.SubprocessError("Command open failed")

    monkeypatch.setattr(subprocess, "run", mock_run)

    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    gptscan.open_file()

    mock_showerror.assert_called_once_with("Error", "Could not open file '/path/to/file.py': Command open failed")


def test_open_file_win32_oserror(monkeypatch):
    monkeypatch.setattr(gptscan, "_resolve_file_paths", lambda evt: ["C:\\path\\file.py"])
    monkeypatch.setattr(sys, "platform", "win32")

    def mock_startfile(path):
        raise OSError("WinError 2 File not found")

    monkeypatch.setattr(os, "startfile", mock_startfile, raising=False)

    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    gptscan.open_file()

    mock_showerror.assert_called_once_with("Error", "Could not open file 'C:\\path\\file.py': WinError 2 File not found")


def test_show_in_folder_oserror(monkeypatch):
    monkeypatch.setattr(gptscan, "_resolve_file_paths", lambda evt: ["/path/to/file.py"])
    monkeypatch.setattr(sys, "platform", "linux")

    def mock_run(*args, **kwargs):
        raise OSError("xlog-open missing")

    monkeypatch.setattr(subprocess, "run", mock_run)

    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    gptscan.show_in_folder()

    mock_showerror.assert_called_once_with("Error", "Could not reveal file '/path/to/file.py': xlog-open missing")


def test_show_in_folder_subprocess_error_win32(monkeypatch):
    monkeypatch.setattr(gptscan, "_resolve_file_paths", lambda evt: ["C:\\path\\file.py"])
    monkeypatch.setattr(sys, "platform", "win32")

    def mock_run(*args, **kwargs):
        raise subprocess.SubprocessError("Explorer launch failed")

    monkeypatch.setattr(subprocess, "run", mock_run)

    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    gptscan.show_in_folder()

    mock_showerror.assert_called_once_with("Error", "Could not reveal file 'C:\\path\\file.py': Explorer launch failed")
