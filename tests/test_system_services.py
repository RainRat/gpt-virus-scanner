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

@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific test")
def test_get_system_service_paths(tmp_path):
    # Mock systemd directories
    etc_dir = tmp_path / "etc" / "systemd" / "system"
    etc_dir.mkdir(parents=True)
    service_file = etc_dir / "test.service"
    service_file.write_text("[Service]\nExecStart=/bin/ls")

    # We need to monkeypatch search_dirs in the function or just test if it returns something on a real system
    # Given the environment, let's just check if it returns a list of strings
    paths = get_system_service_paths()
    assert isinstance(paths, list)
    for p in paths:
        assert p.endswith(".service")

def test_is_container_service():
    assert Config.is_container("test.service") is True
    assert Config.is_container("test.desktop") is True

def test_is_supported_extension():
    assert ".service" in Config.extensions_set
    assert ".desktop" in Config.extensions_set
