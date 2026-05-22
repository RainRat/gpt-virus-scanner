import pytest
import sys
import json
import subprocess
from unittest.mock import patch, MagicMock
import gptscan

def test_get_system_service_commands_windows_multiple():
    mock_json = [
        {"Name": "Service1", "PathName": "C:\\Windows\\System32\\service1.exe"},
        {"Name": "Service2", "PathName": "\"C:\\Program Files\\App\\service2.exe\" --run"}
    ]

    with patch("sys.platform", "win32"), \
         patch("subprocess.check_output") as mock_run:
        mock_run.return_value = json.dumps(mock_json)

        results = gptscan.get_system_service_commands()

        assert len(results) == 2
        assert results[0] == ("[Service] Service1", b"C:\\Windows\\System32\\service1.exe")
        assert results[1] == ("[Service] Service2", b"\"C:\\Program Files\\App\\service2.exe\" --run")

def test_get_system_service_commands_windows_single():
    mock_json = {"Name": "SoloService", "PathName": "C:\\solo.exe"}

    with patch("sys.platform", "win32"), \
         patch("subprocess.check_output") as mock_run:
        mock_run.return_value = json.dumps(mock_json)

        results = gptscan.get_system_service_commands()

        assert len(results) == 1
        assert results[0] == ("[Service] SoloService", b"C:\\solo.exe")

def test_get_system_service_commands_windows_empty():
    with patch("sys.platform", "win32"), \
         patch("subprocess.check_output") as mock_run:
        mock_run.return_value = ""

        results = gptscan.get_system_service_commands()
        assert results == []

def test_get_system_service_commands_windows_error():
    with patch("sys.platform", "win32"), \
         patch("subprocess.check_output") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

        # Should catch exception and return empty list
        results = gptscan.get_system_service_commands()
        assert results == []

def test_get_system_service_commands_non_windows():
    with patch("sys.platform", "linux"):
        results = gptscan.get_system_service_commands()
        assert results == []
