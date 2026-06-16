import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from gptscan import get_desktop_paths, scan_desktop_click, get_system_audit_data

def test_get_desktop_paths():
    """Verify that get_desktop_paths returns a list."""
    paths = get_desktop_paths()
    assert isinstance(paths, list)
    home = Path.home()
    desktop = home / "Desktop"
    if desktop.exists():
        assert str(desktop) in paths
    else:
        assert str(desktop) not in paths

def test_system_audit_includes_desktop():
    """Verify that the system audit data includes desktop folder paths if they exist."""
    all_paths, _ = get_system_audit_data()
    desktop_paths = get_desktop_paths()
    for p in desktop_paths:
        assert p in all_paths

@patch('gptscan._set_scan_target')
@patch('gptscan.button_click')
@patch('gptscan.messagebox.showinfo')
def test_scan_desktop_click(mock_showinfo, mock_button_click, mock_set_target):
    """Verify that scan_desktop_click sets the target and triggers a scan if paths exist."""
    with patch('gptscan.get_desktop_paths', return_value=['/mock/desktop']):
        scan_desktop_click()
        mock_set_target.assert_called_once_with(['/mock/desktop'])
        mock_button_click.assert_called_once()

def test_scan_desktop_click_no_paths():
    """Verify that scan_desktop_click shows info if no paths exist."""
    with patch('gptscan.get_desktop_paths', return_value=[]):
        with patch('gptscan._set_scan_target') as mock_set_target:
            with patch('gptscan.button_click') as mock_button_click:
                with patch('gptscan.messagebox.showinfo') as mock_showinfo:
                    scan_desktop_click()
                    mock_set_target.assert_not_called()
                    mock_button_click.assert_not_called()
                    mock_showinfo.assert_called_once()
