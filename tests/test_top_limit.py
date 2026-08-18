import json
import io
import sys
import unittest
from unittest.mock import patch, MagicMock

import gptscan


class TestTopLimit(unittest.TestCase):
    def test_run_cli_top_limit_csv(self):
        # Mock scan_files yielding 3 results with different threat scores
        def mock_scan_files(*args, **kwargs):
            yield ('result', ('file1.py', '90%', 'Admin notes', 'User notes', '90%', 'import os', '1'))
            yield ('result', ('file2.py', '60%', 'Admin notes', 'User notes', '60%', 'eval(code)', '10'))
            yield ('result', ('file3.py', '80%', 'Admin notes', 'User notes', '80%', 'exec(code)', '5'))
            yield ('summary', (3, 1000, 0.5))

        output_buf = io.StringIO()
        with patch('gptscan.scan_files', side_effect=mock_scan_files), \
             patch('sys.stdout', new=output_buf):
            threats = gptscan.run_cli(['.'], deep=False, show_all=False, use_gpt=False, rate_limit=60, output_format='csv', top_limit=2)

        self.assertEqual(threats, 3) # Total threats count remains 3
        output = output_buf.getvalue()
        lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        # Header + top 2 rows
        self.assertEqual(len(lines), 3)
        self.assertIn("file1.py", lines[1]) # 90% threat highest
        self.assertIn("file3.py", lines[2]) # 80% threat second highest

    def test_run_cli_top_limit_json(self):
        def mock_scan_files(*args, **kwargs):
            yield ('result', ('low.py', '55%', 'Admin', 'User', '55%', 'snippet', '1'))
            yield ('result', ('high.py', '95%', 'Admin', 'User', '95%', 'snippet', '1'))
            yield ('result', ('med.py', '75%', 'Admin', 'User', '75%', 'snippet', '1'))
            yield ('summary', (3, 1000, 0.5))

        output_buf = io.StringIO()
        with patch('gptscan.scan_files', side_effect=mock_scan_files), \
             patch('sys.stdout', new=output_buf):
            threats = gptscan.run_cli(['.'], deep=False, show_all=False, use_gpt=False, rate_limit=60, output_format='json', top_limit=1)

        self.assertEqual(threats, 3)
        output = output_buf.getvalue().strip()
        lines = [line for line in output.splitlines() if line]
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record['path'], 'high.py')

    def test_run_cli_top_limit_larger_than_results(self):
        def mock_scan_files(*args, **kwargs):
            yield ('result', ('only_one.py', '85%', 'Admin', 'User', '85%', 'snippet', '1'))
            yield ('summary', (1, 500, 0.1))

        output_buf = io.StringIO()
        with patch('gptscan.scan_files', side_effect=mock_scan_files), \
             patch('sys.stdout', new=output_buf):
            threats = gptscan.run_cli(['.'], deep=False, show_all=False, use_gpt=False, rate_limit=60, output_format='json', top_limit=10)

        self.assertEqual(threats, 1)
        lines = [line for line in output_buf.getvalue().strip().splitlines() if line]
        self.assertEqual(len(lines), 1)

    def test_cli_top_argument_validation(self):
        # Verify that --top 0 or negative values trigger a parser error
        test_args = ['gptscan.py', '.', '--cli', '--top', '0']
        with patch.object(sys, 'argv', test_args):
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
                    with self.assertRaises(SystemExit):
                        gptscan.main()
                    self.assertIn("Value for --top / --limit must be a positive integer.", mock_stderr.getvalue())

    def test_cli_top_and_limit_alias(self):
        test_args = ['gptscan.py', '.', '--cli', '--limit', '5']
        with patch.object(sys, 'argv', test_args):
            with patch('gptscan.run_cli', return_value=0) as mock_run_cli:
                gptscan.main()
                mock_run_cli.assert_called_once()
                kwargs = mock_run_cli.call_args.kwargs
                self.assertEqual(kwargs.get('top_limit'), 5)


if __name__ == '__main__':
    unittest.main()
