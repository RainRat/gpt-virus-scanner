import os
import unittest
from unittest.mock import patch, MagicMock
from gptscan import get_system_path_dirs, scan_system_path_click

class TestSystemPath(unittest.TestCase):
    @patch('os.environ.get')
    @patch('os.path.isdir')
    @patch('os.path.abspath')
    def test_get_system_path_dirs(self, mock_abspath, mock_isdir, mock_env_get):
        # Setup mock environment
        sep = os.pathsep
        mock_env_get.return_value = f"/bin{sep}/usr/bin{sep}/nonexistent{sep}/bin"

        # Define what is a directory
        def isdir_side_effect(path):
            return path in ["/bin", "/usr/bin"]
        mock_isdir.side_effect = isdir_side_effect

        # Abspath returns the same path for simplicity
        mock_abspath.side_effect = lambda x: x

        dirs = get_system_path_dirs()

        # Should be deduplicated and only exist if isdir is True
        self.assertEqual(dirs, ["/bin", "/usr/bin"])
        self.assertEqual(len(dirs), 2)

    @patch('gptscan.get_system_path_dirs')
    @patch('gptscan._set_scan_target')
    @patch('gptscan.button_click')
    def test_scan_system_path_click_success(self, mock_button_click, mock_set_target, mock_get_dirs):
        mock_get_dirs.return_value = ["/bin", "/usr/bin"]

        scan_system_path_click()

        mock_set_target.assert_called_once_with(["/bin", "/usr/bin"])
        mock_button_click.assert_called_once()

    @patch('gptscan.get_system_path_dirs')
    @patch('gptscan.messagebox.showinfo')
    def test_scan_system_path_click_empty(self, mock_showinfo, mock_get_dirs):
        mock_get_dirs.return_value = []

        scan_system_path_click()

        mock_showinfo.assert_called_once()

if __name__ == '__main__':
    unittest.main()
