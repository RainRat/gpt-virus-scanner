import pytest
from unittest.mock import MagicMock, patch
import gptscan
import shlex
import os

@pytest.fixture
def mock_gui_vars():
    with patch('gptscan.textbox', MagicMock()) as mock_textbox, \
         patch('gptscan.git_var', MagicMock()) as mock_git, \
         patch('gptscan.dry_var', MagicMock()) as mock_dry, \
         patch('gptscan.deep_var', MagicMock()) as mock_deep, \
         patch('gptscan.all_var', MagicMock()) as mock_all, \
         patch('gptscan.gpt_var', MagicMock()) as mock_gpt:

        mock_textbox.get.return_value = ""
        mock_git.get.return_value = False
        mock_dry.get.return_value = False
        mock_deep.get.return_value = False
        mock_all.get.return_value = False
        mock_gpt.get.return_value = False

        yield {
            'textbox': mock_textbox,
            'git': mock_git,
            'dry': mock_dry,
            'deep': mock_deep,
            'all': mock_all,
            'gpt': mock_gpt
        }

def test_set_scan_target_multi():
    mock_textbox = MagicMock()
    with patch('gptscan.textbox', mock_textbox), \
         patch('gptscan.scan_button', MagicMock()):

        # Test single path
        gptscan._set_scan_target("/path/to/file")
        mock_textbox.insert.assert_called_with(0, "/path/to/file")

        mock_textbox.reset_mock()

        # Test single path with spaces
        gptscan._set_scan_target("/path with spaces/folder")
        mock_textbox.insert.assert_called_with(0, "'/path with spaces/folder'")

        mock_textbox.reset_mock()

        # Test multiple paths
        paths = ["/path/one", "/path/two with spaces"]
        gptscan._set_scan_target(paths)
        expected = shlex.join(paths)
        mock_textbox.insert.assert_called_with(0, expected)

def test_button_click_multi_target_parsing(mock_gui_vars):
    paths = ["/path/1", "/path/2 with spaces"]
    mock_gui_vars['textbox'].get.return_value = shlex.join(paths)

    with patch('gptscan.run_scan'), \
         patch('gptscan.clear_results'), \
         patch('gptscan.set_scanning_state'), \
         patch('gptscan.update_status'), \
         patch('gptscan.threading.Thread') as mock_thread:

        gptscan.button_click()

        # Verify scan_targets passed to run_scan (via thread args)
        args, kwargs = mock_thread.call_args
        # kwargs is empty, args is (run_scan, scan_args)
        # scan_args is a tuple, scan_args[0] is scan_targets
        scan_args = kwargs.get('args') or args[1]
        assert scan_args[0] == paths

def test_copy_cli_command_multi_target(mock_gui_vars):
    paths = ["/path/1", "/path/2"]
    mock_gui_vars['textbox'].get.return_value = shlex.join(paths)

    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        command = mock_root.clipboard_append.call_args[0][0]
        assert "/path/1" in command
        assert "/path/2" in command
        # Ensure they are separate arguments
        parts = shlex.split(command)
        assert "/path/1" in parts
        assert "/path/2" in parts

def test_button_click_git_multi_target(mock_gui_vars):
    paths = ["/repo1", "/repo2"]
    mock_gui_vars['textbox'].get.return_value = shlex.join(paths)
    mock_gui_vars['git'].get.return_value = True

    with patch('gptscan.get_git_changed_files') as mock_get_git, \
         patch('gptscan.threading.Thread') as mock_thread, \
         patch('gptscan.clear_results'), \
         patch('gptscan.set_scanning_state'), \
         patch('gptscan.update_status'):

        mock_get_git.side_effect = [["/repo1/a.py"], ["/repo2/b.py"]]

        gptscan.button_click()

        assert mock_get_git.call_count == 2
        args, kwargs = mock_thread.call_args
        scan_args = kwargs.get('args') or args[1]
        assert set(scan_args[0]) == {"/repo1/a.py", "/repo2/b.py"}
