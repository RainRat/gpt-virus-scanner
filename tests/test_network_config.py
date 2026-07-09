import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from gptscan import get_network_config_paths

def test_get_network_config_paths_linux():
    with patch('sys.platform', 'linux'), \
         patch('os.path.isfile', side_effect=lambda p: p in ['/etc/hosts', '/etc/resolv.conf']), \
         patch('os.path.isdir', return_value=False):
        paths = get_network_config_paths()
        # Paths are returned as absolute strings
        assert any(p.endswith('/etc/hosts') for p in paths)
        assert any(p.endswith('/etc/resolv.conf') for p in paths)

def test_get_network_config_paths_windows():
    with patch('sys.platform', 'win32'), \
         patch('os.environ.get', return_value='C:\\Windows'), \
         patch('os.path.isfile', side_effect=lambda p: 'hosts' in p):
        paths = get_network_config_paths()
        # On Linux runner, Path(p).absolute() might use forward slashes even if we mock win32
        assert any('hosts' in p for p in paths)
