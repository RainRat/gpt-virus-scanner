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

def test_get_system_service_commands_linux(tmp_path):
    # Mock systemd service files
    service_dir = tmp_path / "systemd"
    service_dir.mkdir()

    service_file = service_dir / "test.service"
    service_file.write_text(
        "[Unit]\nDescription=Test\n"
        "[Service]\n"
        "ExecStart=/usr/bin/test-cmd --arg1 \\\n  --arg2\n"
        "ExecStartPost=/usr/bin/post-cmd\n"
        "Environment=FOO=BAR\n"
    )

    with patch("sys.platform", "linux"), \
         patch("gptscan.get_system_service_paths", return_value=[str(service_file)]):

        results = gptscan.get_system_service_commands()

        assert len(results) == 2
        # Check multiline command
        assert results[0] == ("[Service] test.service", b"/usr/bin/test-cmd --arg1 --arg2")
        # Check single line command
        assert results[1] == ("[Service] test.service", b"/usr/bin/post-cmd")

def test_get_system_service_commands_non_supported():
    with patch("sys.platform", "darwin"):
        results = gptscan.get_system_service_commands()
        assert results == []
