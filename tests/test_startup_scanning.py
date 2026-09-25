import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import json

class TestStartupScanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Save original modules
        cls.orig_modules = {}
        mock_mods = [
            'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 
            'tkinter.filedialog', 'tkinter.simpledialog', 
            'tkinter.scrolledtext', 'tkinter.font'
        ]
        for mod in mock_mods:
            if mod in sys.modules:
                cls.orig_modules[mod] = sys.modules[mod]
            
            m = MagicMock()
            # Add __code__ to mainloop to satisfy matplotlib inspection if needed
            if hasattr(m, 'mainloop'):
                m.mainloop.__code__ = (lambda: None).__code__
            sys.modules[mod] = m
        
        # Now import gptscan
        global gptscan
        if 'gptscan' in sys.modules:
            import importlib
            importlib.reload(sys.modules['gptscan'])
        import gptscan

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        mock_mods = [
            'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 
            'tkinter.filedialog', 'tkinter.simpledialog', 
            'tkinter.scrolledtext', 'tkinter.font'
        ]
        for mod in mock_mods:
            if mod in cls.orig_modules:
                sys.modules[mod] = cls.orig_modules[mod]
            else:
                del sys.modules[mod]

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

    @patch('gptscan.subprocess.check_output')
    @patch('gptscan.sys.platform', 'win32')
    def test_get_startup_item_commands_windows_single_dict(self, mock_check_output):
        mock_output = json.dumps({"Name": "SingleApp", "Command": "C:\\single.exe"})
        mock_check_output.return_value = mock_output

        results = gptscan.get_startup_item_commands()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "[Startup] SingleApp")
        self.assertEqual(results[0][1], b"C:\\single.exe")

    @patch('gptscan.Path.exists')
    @patch('gptscan.Path.glob')
    @patch('gptscan.sys.platform', 'linux')
    def test_get_startup_item_commands_linux(self, mock_glob, mock_exists):
        mock_exists.side_effect = lambda: True

        # Mock .desktop file paths
        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "test.desktop"
        
        mock_glob.return_value = [mock_file1]

        with patch('gptscan.open', mock_open(read_data="[Desktop Entry]\nExec=test-cmd --start\n")):
            results = gptscan.get_startup_item_commands()

        self.assertEqual(len(results), 2) # Two search dirs
        self.assertEqual(results[0][0], "[Autostart] test.desktop")
        self.assertEqual(results[0][1], b"test-cmd --start")

    @patch('gptscan.subprocess.check_output')
    @patch('gptscan.sys.platform', 'win32')
    def test_get_startup_item_commands_windows_exception(self, mock_check_output):
        import subprocess
        mock_check_output.side_effect = subprocess.CalledProcessError(1, ["powershell"])
        results = gptscan.get_startup_item_commands()
        self.assertEqual(results, [])

    @patch('gptscan.Path.exists')
    @patch('gptscan.Path.glob')
    @patch('gptscan.sys.platform', 'linux')
    def test_get_startup_item_commands_linux_missing_exec_and_oserror(self, mock_glob, mock_exists):
        mock_exists.return_value = True

        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "noexec.desktop"
        mock_file2 = MagicMock(spec=Path)
        mock_file2.name = "error.desktop"

        mock_glob.return_value = [mock_file1, mock_file2]

        def custom_open(path, *args, **kwargs):
            if "error.desktop" in str(path):
                raise OSError("Read error")
            return mock_open(read_data="[Desktop Entry]\nComment=No Exec line here\n")()

        with patch('gptscan.open', side_effect=custom_open):
            results = gptscan.get_startup_item_commands()

        self.assertEqual(results, [])

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

    @patch('gptscan.Path.exists')
    @patch('gptscan.Path.glob')
    @patch('gptscan.plistlib.load')
    @patch('gptscan.sys.platform', 'darwin')
    def test_get_startup_item_commands_macos_program_key_and_string_args(self, mock_plist_load, mock_glob, mock_exists):
        mock_exists.return_value = True

        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "com.program.plist"
        mock_file2 = MagicMock(spec=Path)
        mock_file2.name = "com.stringargs.plist"
        mock_file3 = MagicMock(spec=Path)
        mock_file3.name = "com.corrupt.plist"

        mock_glob.return_value = [mock_file1, mock_file2, mock_file3]

        plist_data = {
            "com.program.plist": {"Program": "/usr/bin/program-exec"},
            "com.stringargs.plist": {"ProgramArguments": "/usr/bin/single-arg"},
            "com.corrupt.plist": None
        }

        def mock_open_side_effect(path, *args, **kwargs):
            filename = getattr(path, 'name', str(path))
            if filename in plist_data and plist_data[filename] is None:
                raise OSError("Corrupt file")
            m = mock_open()()
            m.filename = filename
            return m

        def side_effect_plist(f):
            filename = getattr(f, 'filename', '')
            if filename in plist_data and plist_data[filename] is not None:
                return plist_data[filename]
            import plistlib
            raise plistlib.InvalidFileException("Corrupt plist")

        mock_plist_load.side_effect = side_effect_plist

        with patch('gptscan.open', side_effect=mock_open_side_effect):
            results = gptscan.get_startup_item_commands()

        # Each search dir processes the glob list
        items = [r for r in results if r[0].startswith("[LaunchAgent]")]
        self.assertTrue(len(items) >= 2)
        prog_item = next(r for r in items if "com.program.plist" in r[0])
        str_item = next(r for r in items if "com.stringargs.plist" in r[0])
        self.assertEqual(prog_item[1], b"/usr/bin/program-exec")
        self.assertEqual(str_item[1], b"/usr/bin/single-arg")

if __name__ == '__main__':
    unittest.main()
