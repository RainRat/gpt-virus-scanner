import pytest
from pathlib import Path
from unittest.mock import MagicMock
import sys
import os
from gptscan import unpack_content, get_system_service_paths, Config

def test_unpack_service_file():
    content = b"""
[Unit]
Description=Test Service

[Service]
ExecStart=/usr/bin/python3 /tmp/script.py
ExecStop=/usr/bin/killall python3
Restart=always

[Install]
WantedBy=multi-user.target
"""
    snippets = list(unpack_content("test.service", content))
    assert len(snippets) == 2
    assert snippets[0][0] == "test.service [Command 1]"
    assert b"/usr/bin/python3 /tmp/script.py" in snippets[0][1]
    assert snippets[1][0] == "test.service [Command 2]"
    assert b"/usr/bin/killall python3" in snippets[1][1]

def test_unpack_service_multiline():
    content = b"""
[Service]
ExecStart=/usr/bin/python3 \\
    /tmp/script.py \\
    --arg1
"""
    snippets = list(unpack_content("test.service", content))
    assert len(snippets) == 1
    assert b"/usr/bin/python3 /tmp/script.py --arg1" in snippets[0][1]

def test_unpack_desktop_file():
    content = b"""
[Desktop Entry]
Name=Test
Exec=/usr/bin/test-app --flag
Type=Application
"""
    snippets = list(unpack_content("test.desktop", content))
    assert len(snippets) == 1
    assert b"/usr/bin/test-app --flag" in snippets[0][1]

def test_get_system_service_paths_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert get_system_service_paths() == []

def test_get_system_service_paths_linux_mocked(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    mock_file = MagicMock(spec=Path)
    mock_file.is_file.return_value = True
    mock_file.is_symlink.return_value = False
    mock_file.__str__.return_value = "/etc/systemd/system/good.service"

    mock_symlink = MagicMock(spec=Path)
    mock_symlink.is_file.return_value = True
    mock_symlink.is_symlink.return_value = True
    mock_symlink.__str__.return_value = "/etc/systemd/system/symlink.service"

    mock_error = MagicMock(spec=Path)
    mock_error.is_file.side_effect = OSError("Permission denied")
    mock_error.__str__.return_value = "/etc/systemd/system/error.service"

    def mock_rglob(self, pattern):
        path_str = str(self).replace("\\", "/")
        if "etc" in path_str:
            return [mock_file, mock_symlink, mock_error]
        return []

    monkeypatch.setattr(Path, "rglob", mock_rglob)
    monkeypatch.setattr(Path, "home", lambda: Path("/mock/home"))

    results = get_system_service_paths()
    assert results == ["/etc/systemd/system/good.service"]

def test_is_container_service():
    assert Config.is_container("test.service") is True
    assert Config.is_container("test.desktop") is True

def test_is_supported_extension():
    assert ".service" in Config.extensions_set
    assert ".desktop" in Config.extensions_set
