import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import _set_scan_target, button_click, copy_cli_command
import shlex

@pytest.fixture
def mock_gui_globals():
    """Mock the global GUI variables in gptscan."""
    with patch('gptscan.textbox', MagicMock()) as mock_textbox, \
         patch('gptscan.scan_button', MagicMock()) as mock_scan_btn, \
         patch('gptscan.git_var', MagicMock()) as mock_git_var, \
         patch('gptscan.root', MagicMock()) as mock_root:

        # Set default values for mocks
        mock_git_var.get.return_value = False

        yield {
            'textbox': mock_textbox,
            'scan_button': mock_scan_btn,
            'git_var': mock_git_var,
            'root': mock_root
        }

def test_set_scan_target_multi(mock_gui_globals):
    """Verify that multiple paths are correctly joined in the textbox."""
    paths = ["file1.py", "/path with spaces/file2.js"]
    _set_scan_target(paths)

    # Verify shlex.join was used to format the paths
    expected = shlex.join(paths)
    mock_gui_globals['textbox'].insert.assert_called_with(0, expected)

def test_button_click_parsing(mock_gui_globals):
    """Verify that button_click correctly parses multiple targets using shlex.split."""
    targets = ["file1.py", "file2.js"]
    raw_path = shlex.join(targets)
    mock_gui_globals['textbox'].get.return_value = raw_path

    with patch('gptscan.run_scan') as mock_run_scan, \
         patch('gptscan.clear_results'), \
         patch('gptscan.set_scanning_state'), \
         patch('gptscan.update_status'), \
         patch('gptscan.dry_var', MagicMock(get=lambda: True)), \
         patch('gptscan.deep_var', MagicMock(get=lambda: False)), \
         patch('gptscan.all_var', MagicMock(get=lambda: False)), \
         patch('gptscan.gpt_var', MagicMock(get=lambda: False)):

        button_click()

        # The first argument to run_scan should be the list of parsed targets
        args, _ = mock_run_scan.call_args
        assert args[0] == targets

def test_copy_cli_command_multi_target(mock_gui_globals):
    """Verify that copy_cli_command handles multiple targets correctly."""
    targets = ["file1.py", "/path with spaces/file2.js"]
    raw_path = shlex.join(targets)
    mock_gui_globals['textbox'].get.return_value = raw_path

    # Mocking deep_var and other globals required for copy_cli_command
    with patch('gptscan.deep_var', MagicMock(get=lambda: False)), \
         patch('gptscan.git_var', MagicMock(get=lambda: False)), \
         patch('gptscan.dry_var', MagicMock(get=lambda: False)), \
         patch('gptscan.all_var', MagicMock(get=lambda: False)), \
         patch('gptscan.scan_all_var', MagicMock(get=lambda: False)), \
         patch('gptscan.gpt_var', MagicMock(get=lambda: False)), \
         patch('gptscan.update_status'):

        copy_cli_command()

        # Verify CLI command includes individually quoted targets
        command = mock_gui_globals['root'].clipboard_append.call_args[0][0]
        assert "python gptscan.py file1.py '/path with spaces/file2.js' --cli" in command

def test_button_click_malformed_input(mock_gui_globals):
    """Verify that malformed input is handled with an error message."""
    mock_gui_globals['textbox'].get.return_value = 'file1.py "unclosed quote'

    with patch('gptscan.messagebox.showerror') as mock_error, \
         patch('gptscan.clear_results'):
        button_click()
        mock_error.assert_called_once()
        assert "Malformed path selection" in mock_error.call_args[0][1]
