import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Mocking modules that might not be available or are not needed for this unit test
sys.modules['tensorflow'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['keras.models'] = MagicMock()

import gptscan

def test_get_python_package_paths():
    with patch('site.getsitepackages') as mock_getsite:
        with patch('site.getusersitepackages') as mock_getuser:
            with patch('os.path.exists') as mock_exists:
                mock_getsite.return_value = ['/global/site-packages']
                mock_getuser.return_value = '/user/site-packages'
                mock_exists.side_effect = lambda p: p in ['/global/site-packages', '/user/site-packages']

                paths = gptscan.get_python_package_paths()
                assert '/global/site-packages' in paths
                assert '/user/site-packages' in paths

def test_system_audit_integration():
    with patch('gptscan.get_python_package_paths') as mock_get_pkg:
        mock_get_pkg.return_value = ['/fake/site-packages']
        paths, snippets = gptscan.get_system_audit_data()
        assert '/fake/site-packages' in paths

def test_cli_python_packages_flag():
    with patch('gptscan.get_python_package_paths') as mock_get_pkg:
        with patch('gptscan.run_cli') as mock_run_cli:
            mock_get_pkg.return_value = ['/fake/site-packages']

            # Simulate CLI call
            test_args = ['gptscan.py', '--python-packages', '--cli']
            with patch.object(sys, 'argv', test_args):
                 gptscan.main()

            assert mock_get_pkg.called
            # Verify that the package path was added to scan_targets passed to run_cli
            args, kwargs = mock_run_cli.call_args
            assert '/fake/site-packages' in args[0]
