import pytest
from pathlib import Path
from gptscan import unpack_content, get_system_service_paths, Config
import os
import sys

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

    monkeypatch.setattr(sys, "platform", "darwin")
    assert get_system_service_paths() == []

def test_get_system_service_paths_linux_mocked(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    mock_home = Path("/home/mockuser")
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    class MockServicePath:
        def __init__(self, path_str, is_file_val=True, is_symlink_val=False, raise_exc=False):
            self.path_str = path_str
            self.is_file_val = is_file_val
            self.is_symlink_val = is_symlink_val
            self.raise_exc = raise_exc

        def is_file(self):
            if self.raise_exc:
                raise OSError("Access denied")
            return self.is_file_val

        def is_symlink(self):
            return self.is_symlink_val

        def __str__(self):
            return self.path_str

    mock_paths = [
        MockServicePath("/etc/systemd/system/z_service.service", is_file_val=True, is_symlink_val=False),
        MockServicePath("/etc/systemd/system/a_service.service", is_file_val=True, is_symlink_val=False),
        MockServicePath("/lib/systemd/system/symlink.service", is_file_val=True, is_symlink_val=True),
        MockServicePath("/usr/lib/systemd/system/error.service", raise_exc=True),
        MockServicePath("/etc/systemd/system/a_service.service", is_file_val=True, is_symlink_val=False),
    ]

    concrete_path_cls = type(Path())
    monkeypatch.setattr(concrete_path_cls, "rglob", lambda self, pattern: mock_paths if pattern == "*.service" else [])

    results = get_system_service_paths()

    assert results == [
        "/etc/systemd/system/a_service.service",
        "/etc/systemd/system/z_service.service"
    ]

def test_is_container_service():
    assert Config.is_container("test.service") is True
    assert Config.is_container("test.desktop") is True

def test_is_supported_extension():
    assert ".service" in Config.extensions_set
    assert ".desktop" in Config.extensions_set
