import pytest
from unittest.mock import MagicMock, patch
import gptscan
import shlex

def test_set_scan_target_multi():
    with patch('gptscan.textbox') as mock_textbox, \
         patch('gptscan.scan_button') as mock_button:
        # Test single path
        gptscan._set_scan_target("/path/to/file")
        mock_textbox.delete.assert_called_with(0, gptscan.tk.END)
        mock_textbox.insert.assert_called_with(0, shlex.quote("/path/to/file"))

        # Test multiple paths
        gptscan._set_scan_target(["/path one", "/path two"])
        mock_textbox.insert.assert_called_with(0, shlex.join(["/path one", "/path two"]))

def test_button_click_multi_target():
    with patch('gptscan.textbox') as mock_textbox, \
         patch('gptscan.git_var') as mock_git, \
         patch('gptscan.dry_var') as mock_dry, \
         patch('gptscan.deep_var') as mock_deep, \
         patch('gptscan.all_var') as mock_all, \
         patch('gptscan.gpt_var') as mock_gpt, \
         patch('gptscan.threading.Thread') as mock_thread, \
         patch('gptscan.os.path.exists', return_value=True), \
         patch('gptscan.set_scanning_state'), \
         patch('gptscan.update_status'), \
         patch('gptscan.clear_results'):

        mock_textbox.get.return_value = "'/path one' '/path two'"
        mock_git.get.return_value = False
        mock_dry.get.return_value = False
        mock_deep.get.return_value = False
        mock_all.get.return_value = False
        mock_gpt.get.return_value = False
        gptscan.current_cancel_event = None

        gptscan.button_click()

        # Check if run_scan was called with the split targets
        args, kwargs = mock_thread.call_args
        scan_args = kwargs.get('args') or args[0] # target, deep, show_all, use_gpt, cancel_event, ...
        assert scan_args[0] == ["/path one", "/path two"]

def test_button_click_invalid_format():
    with patch('gptscan.textbox') as mock_textbox, \
         patch('gptscan.messagebox.showerror') as mock_error, \
         patch('gptscan.clear_results'):

        mock_textbox.get.return_value = "'unclosed quote"
        gptscan.current_cancel_event = None

        gptscan.button_click()
        mock_error.assert_called_once()
        assert "Input Error" in mock_error.call_args[0][0]

def test_copy_cli_command_multi():
    with patch('gptscan.textbox') as mock_textbox, \
         patch('gptscan.root') as mock_root, \
         patch('gptscan.Config') as mock_config, \
         patch('gptscan.deep_var') as mock_deep, \
         patch('gptscan.git_var') as mock_git, \
         patch('gptscan.dry_var') as mock_dry, \
         patch('gptscan.all_var') as mock_all, \
         patch('gptscan.scan_all_var') as mock_scan_all, \
         patch('gptscan.gpt_var') as mock_gpt:

        mock_textbox.get.return_value = "target1 \"target two\""
        mock_config.THRESHOLD = 50
        mock_config.provider = "openai"
        mock_deep.get.return_value = False
        mock_git.get.return_value = False
        mock_dry.get.return_value = False
        mock_all.get.return_value = False
        mock_scan_all.get.return_value = False
        mock_gpt.get.return_value = False

        gptscan.copy_cli_command()

        # Verify the command in clipboard
        args, kwargs = mock_root.clipboard_append.call_args
        command = args[0]
        assert "target1" in command
        assert "'target two'" in command
        assert "--cli" in command

def test_button_click_git_multi():
    with patch('gptscan.textbox') as mock_textbox, \
         patch('gptscan.git_var') as mock_git, \
         patch('gptscan.dry_var') as mock_dry, \
         patch('gptscan.deep_var') as mock_deep, \
         patch('gptscan.all_var') as mock_all, \
         patch('gptscan.gpt_var') as mock_gpt, \
         patch('gptscan.get_git_changed_files') as mock_get_git, \
         patch('gptscan.threading.Thread') as mock_thread, \
         patch('gptscan.os.path.exists', return_value=True), \
         patch('gptscan.set_scanning_state'), \
         patch('gptscan.update_status'), \
         patch('gptscan.clear_results'):

        mock_textbox.get.return_value = "dir1 dir2"
        mock_git.get.return_value = True
        mock_dry.get.return_value = False
        mock_deep.get.return_value = False
        mock_all.get.return_value = False
        mock_gpt.get.return_value = False
        mock_get_git.side_effect = [["dir1/f1.py"], ["dir2/f2.js"]]
        gptscan.current_cancel_event = None

        gptscan.button_click()

        args, kwargs = mock_thread.call_args
        scan_args = kwargs.get('args') or args[0]
        assert set(scan_args[0]) == {"dir1/f1.py", "dir2/f2.js"}
