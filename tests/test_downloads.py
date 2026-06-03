import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys
import os
import gptscan

class TestDownloads(unittest.TestCase):

    @patch("gptscan.Path")
    @patch("gptscan._normalize_and_filter_dirs")
    def test_get_downloads_path_standard(self, mock_normalize, mock_path_class):
        mock_home = MagicMock()
        mock_path_class.home.return_value = mock_home
        # home / "Downloads"
        mock_downloads = MagicMock()
        mock_home.__truediv__.return_value = mock_downloads

        mock_normalize.side_effect = lambda x: [str(p) for p in x]

        with patch("sys.platform", "linux"):
            paths = gptscan.get_downloads_path()
            self.assertIn(str(mock_downloads), paths)

    @patch("gptscan.Path")
    @patch("gptscan._normalize_and_filter_dirs")
    def test_get_downloads_path_xdg(self, mock_normalize, mock_path_class):
        mock_home = MagicMock()
        mock_path_class.home.return_value = mock_home

        # mock home / "Downloads"
        mock_std_downloads = MagicMock()
        mock_std_downloads.__str__.return_value = "/home/user/Downloads"

        # mock home / ".config" / "user-dirs.dirs"
        mock_config_file = MagicMock()
        mock_config_file.exists.return_value = True
        mock_config_file.__str__.return_value = "/home/user/.config/user-dirs.dirs"

        # mock home / "MyDownloads"
        mock_xdg_downloads = MagicMock()
        mock_xdg_downloads.__str__.return_value = "/home/user/MyDownloads"

        def truediv_side_effect(other):
            if other == "Downloads": return mock_std_downloads
            if other == ".config":
                res = MagicMock()
                res.__truediv__.side_effect = lambda x: mock_config_file if x == "user-dirs.dirs" else MagicMock()
                return res
            if other == "MyDownloads": return mock_xdg_downloads
            return MagicMock()

        mock_home.__truediv__.side_effect = truediv_side_effect

        mock_normalize.side_effect = lambda x: [str(p) for p in x]

        m_open = mock_open(read_data='XDG_DOWNLOAD_DIR="$HOME/MyDownloads"\n')

        with patch("sys.platform", "linux"):
            with patch("builtins.open", m_open):
                paths = gptscan.get_downloads_path()
                self.assertIn("/home/user/MyDownloads", paths)

    @patch("gptscan.Path")
    @patch("gptscan._normalize_and_filter_dirs")
    def test_get_downloads_path_windows(self, mock_normalize, mock_path_class):
        mock_home = MagicMock()
        mock_path_class.home.return_value = mock_home
        mock_downloads = MagicMock()
        mock_home.__truediv__.return_value = mock_downloads

        mock_normalize.side_effect = lambda x: [str(p) for p in x]

        with patch("sys.platform", "win32"):
            paths = gptscan.get_downloads_path()
            self.assertIn(str(mock_downloads), paths)

if __name__ == "__main__":
    unittest.main()
