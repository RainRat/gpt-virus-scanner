
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

# Mock tkinter before importing gptscan
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.font'] = MagicMock()

import gptscan

class TestCLIHTML(unittest.TestCase):
    def setUp(self):
        # Redirect stdout and stderr
        self.held_stdout = StringIO()
        self.held_stderr = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.held_stdout
        sys.stderr = self.held_stderr

    def tearDown(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    @patch('gptscan.scan_files')
    def test_run_cli_html(self, mock_scan_files):
        # Setup mock return values for scan_files
        # Yield a progress update then a result
        mock_scan_files.return_value = iter([
            ('progress', (0, 1, 'Scanning...')),
            ('result', (
                'test_file.py',
                '95%',
                'Admin note',
                'User note',
                '99%',
                'import os\nos.system("rm -rf /")'
            )),
            ('progress', (1, 1, None))
        ])

        # Run CLI with html format
        gptscan.run_cli(
            targets=['test_file.py'],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='html'
        )

        output = self.held_stdout.getvalue()

        # Verify HTML structure
        self.assertIn('<!DOCTYPE html>', output)
        self.assertIn('<html lang="en">', output)
        self.assertIn('GPT Scan Results', output)
        self.assertIn('test_file.py', output)
        self.assertIn('high-risk', output) # 99% > 80%
        self.assertIn('99%', output)
        self.assertIn('import os', output)
        self.assertIn('&quot;rm -rf /&quot;', output) # Escaping check

    def test_generate_html_escaping(self):
        results = [{
            "path": "<script>alert(1)</script>",
            "own_conf": "10%",
            "gpt_conf": "",
            "admin_desc": "<b>Bold</b>",
            "end-user_desc": "",
            "snippet": "print('<Hello>')"
        }]

        html_out = gptscan.generate_html(results)

        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)
        self.assertNotIn("<b>Bold</b>", html_out)
        self.assertIn("&lt;b&gt;Bold&lt;/b&gt;", html_out)
        self.assertIn("&lt;Hello&gt;", html_out)

if __name__ == '__main__':
    unittest.main()
