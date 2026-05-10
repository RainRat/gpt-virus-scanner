import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import json
import io

# Mocking modules that might not be fully available or are GUI-related
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.simpledialog'] = MagicMock()
sys.modules['tkinter.scrolledtext'] = MagicMock()
sys.modules['tkinter.font'] = MagicMock()

import gptscan

class TestStartupScanning(unittest.TestCase):

    @patch('gptscan.subprocess.check_output')
    @patch('gptscan.sys.platform', 'win32')
    def test_get_startup_item_commands_windows(self, mock_check_output):
        mock_output = json.dumps([
            {"Name": "TestApp", "Command": "C:\\test.exe"},
            {"Name": "Malicious", "Command": "powershell.exe -enc XXX"}
        ])
        mock_check_output.return_value = mock_output

        results = gptscan.get_startup_item_commands()

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "[Startup] TestApp")
        self.assertEqual(results[0][1], b"C:\\test.exe")
        self.assertEqual(results[1][0], "[Startup] Malicious")

    @patch('gptscan.Path.exists')
    @patch('gptscan.Path.glob')
    @patch('gptscan.sys.platform', 'linux')
    def test_get_startup_item_commands_linux(self, mock_glob, mock_exists):
        mock_exists.side_effect = lambda: True

        # Mock .desktop file paths
        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "test.desktop"
        mock_file1.open = mock_open(read_data="[Desktop Entry]\nExec=test-cmd --start\n")

        mock_glob.return_value = [mock_file1]

        # We need to mock 'with open(p, "r", ...)' inside gptscan.
        # Since it uses Path objects, let's patch builtins.open instead or mock Path.open
        with patch('gptscan.open', mock_open(read_data="[Desktop Entry]\nExec=test-cmd --start\n")):
            results = gptscan.get_startup_item_commands()

        self.assertEqual(len(results), 2) # Two search dirs, same mock file
        self.assertEqual(results[0][0], "[Autostart] test.desktop")
        self.assertEqual(results[0][1], b"test-cmd --start")

    @patch('gptscan.Path.exists')
    @patch('gptscan.Path.glob')
    @patch('gptscan.plistlib.load')
    @patch('gptscan.sys.platform', 'darwin')
    def test_get_startup_item_commands_macos(self, mock_plist_load, mock_glob, mock_exists):
        mock_exists.side_effect = lambda: True

        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "com.test.plist"
        mock_glob.return_value = [mock_file1]

        mock_plist_load.return_value = {"ProgramArguments": ["/usr/bin/test", "-v"]}

        with patch('gptscan.open', mock_open()):
            results = gptscan.get_startup_item_commands()

        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0][0], "[LaunchAgent] com.test.plist")
        self.assertEqual(results[0][1], b"/usr/bin/test -v")

if __name__ == '__main__':
    unittest.main()
