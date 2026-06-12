import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from gptscan import get_temp_paths, scan_temp_click, get_system_audit_data

def test_get_temp_paths():
    """Verify that get_temp_paths returns valid directories and includes the standard temp dir."""
    paths = get_temp_paths()
    assert isinstance(paths, list)
    # tempfile.gettempdir() should always be in there if it exists
    standard_temp = os.path.abspath(tempfile.gettempdir())
    if os.path.isdir(standard_temp):
        assert standard_temp in paths

def test_system_audit_includes_temp():
    """Verify that the system audit data includes temporary folder paths."""
    all_paths, _ = get_system_audit_data()
    temp_paths = get_temp_paths()
    for p in temp_paths:
        assert p in all_paths

@patch('gptscan._set_scan_target')
@patch('gptscan.button_click')
def test_scan_temp_click(mock_button_click, mock_set_target):
    """Verify that scan_temp_click sets the target and triggers a scan."""
    scan_temp_click()
    mock_set_target.assert_called_once()
    mock_button_click.assert_called_once()

    # Check that the paths passed to _set_scan_target are what we expect
    args, _ = mock_set_target.call_args
    assert args[0] == get_temp_paths()
