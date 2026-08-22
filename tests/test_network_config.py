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

def test_get_network_config_paths_linux_config_dirs():
    file_netplan = Path('/etc/netplan/01-netcfg.yaml')
    file_interfaces = Path('/etc/network/interfaces.d/eth0')

    def mock_is_dir(self):
        return str(self) in ['/etc/netplan', '/etc/network/interfaces.d']

    def mock_iterdir(self):
        if str(self) == '/etc/netplan':
            return [file_netplan]
        if str(self) == '/etc/network/interfaces.d':
            return [file_interfaces]
        return []

    def mock_is_file(self):
        return self in [file_netplan, file_interfaces]

    with patch('sys.platform', 'linux'), \
         patch.object(Path, 'is_dir', mock_is_dir), \
         patch.object(Path, 'iterdir', mock_iterdir), \
         patch.object(Path, 'is_file', mock_is_file), \
         patch('os.path.isfile', side_effect=lambda p: p in ['/etc/hosts', str(file_netplan), str(file_interfaces)]):
        paths = get_network_config_paths()
        assert any(p.endswith('01-netcfg.yaml') for p in paths)
        assert any(p.endswith('eth0') for p in paths)

def test_get_network_config_paths_linux_oserror():
    def mock_is_dir(self):
        return str(self) == '/etc/netplan'

    def mock_iterdir(self):
        raise OSError("Permission denied")

    with patch('sys.platform', 'linux'), \
         patch.object(Path, 'is_dir', mock_is_dir), \
         patch.object(Path, 'iterdir', mock_iterdir), \
         patch('os.path.isfile', side_effect=lambda p: p == '/etc/hosts'):
        paths = get_network_config_paths()
        assert any(p.endswith('/etc/hosts') for p in paths)
