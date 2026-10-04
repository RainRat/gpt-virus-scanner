import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import gptscan


class TestIncludePatterns(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root_path = self.temp_dir.name

        # Create nested file structure
        # /
        # ├── src/
        # │   ├── app.py
        # │   └── helper.js
        # ├── tests/
        # │   └── test_app.py
        # └── docs/
        #     └── readme.txt
        os.makedirs(os.path.join(self.root_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.root_path, "tests"), exist_ok=True)
        os.makedirs(os.path.join(self.root_path, "docs"), exist_ok=True)

        self.file_app_py = os.path.join(self.root_path, "src", "app.py")
        self.file_helper_js = os.path.join(self.root_path, "src", "helper.js")
        self.file_test_py = os.path.join(self.root_path, "tests", "test_app.py")
        self.file_readme = os.path.join(self.root_path, "docs", "readme.txt")

        for fpath in [self.file_app_py, self.file_helper_js, self.file_test_py, self.file_readme]:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("print('hello')\n")

    def test_scan_files_include_extension(self):
        # Scan with include_patterns=["*.js"]
        events = list(gptscan.scan_files(
            scan_targets=self.root_path,
            deep_scan=False,
            show_all=True,
            use_gpt=False,
            dry_run=True,
            include_patterns=["*.js"]
        ))
        paths = [data[0] for ev_type, data in events if ev_type == 'result']
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("helper.js"))

    def test_scan_files_include_directory(self):
        # Scan with include_patterns=["src/*"]
        events = list(gptscan.scan_files(
            scan_targets=self.root_path,
            deep_scan=False,
            show_all=True,
            use_gpt=False,
            dry_run=True,
            include_patterns=["src/*"]
        ))
        paths = [data[0] for ev_type, data in events if ev_type == 'result']
        self.assertEqual(len(paths), 2)
        self.assertTrue(all("src" in p for p in paths))

    def test_scan_files_include_and_exclude(self):
        # Scan with include_patterns=["*.py"] and exclude_patterns=["tests/*"]
        events = list(gptscan.scan_files(
            scan_targets=self.root_path,
            deep_scan=False,
            show_all=True,
            use_gpt=False,
            dry_run=True,
            include_patterns=["*.py"],
            exclude_patterns=["tests/*"]
        ))
        paths = [data[0] for ev_type, data in events if ev_type == 'result']
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("app.py"))

    def test_run_cli_with_include_patterns(self):
        # Test run_cli passing include_patterns
        with patch("sys.stdout"):
            count = gptscan.run_cli(
                targets=self.root_path,
                deep=False,
                show_all=True,
                use_gpt=False,
                rate_limit=60,
                dry_run=True,
                include_patterns=["*.txt"],
                quiet=True
            )
            # Dry run returns 0 threats found, but verifies execution
            self.assertEqual(count, 0)

    def test_cli_main_include_flag(self):
        # Test CLI argument parsing for --include
        test_args = ["gptscan.py", self.root_path, "--cli", "--dry-run", "--include", "*.js"]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout"):
                try:
                    gptscan.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 0)


if __name__ == "__main__":
    unittest.main()
