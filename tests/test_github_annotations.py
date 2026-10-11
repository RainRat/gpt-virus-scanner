import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import gptscan


class TestGitHubAnnotations(unittest.TestCase):

    def test_generate_github_annotations_basic(self):
        results = [
            {
                "path": "src/danger.py",
                "line": "12",
                "own_conf": "85%",
                "admin_desc": "High risk eval found",
                "end-user_desc": "Evaluates raw input",
                "snippet": "eval(user_input)"
            },
            {
                "path": "[URL] https://github.com/user/repo/blob/main/low.py",
                "line": "-",
                "own_conf": "30%",
                "admin_desc": "Low risk pattern",
                "end-user_desc": "",
                "snippet": ""
            }
        ]
        output = gptscan.generate_github_annotations(results)
        lines = output.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("::error file=src/danger.py,line=12,title=GPT Scan 85%25 Threat Detected::High risk eval found | Evaluates raw input"))
        self.assertTrue(lines[1].startswith("::warning file=https://github.com/user/repo/blob/main/low.py,line=1,title=GPT Scan 30%25 Threat Detected::Low risk pattern"))

    def test_generate_github_annotations_escaping(self):
        results = [
            {
                "path": "dir/file%name.py",
                "line": "5",
                "own_conf": "90%",
                "admin_desc": "Line 1\nLine 2\r\n100% dangerous",
                "end-user_desc": "Check\nnow"
            }
        ]
        output = gptscan.generate_github_annotations(results)
        self.assertIn("%25", output)
        self.assertIn("%0A", output)
        self.assertNotIn("\nLine 2", output)

    def test_parse_github_annotations_content(self):
        raw_annotations = (
            "::error file=src/danger.py,line=12,title=GPT Scan 85%25 Threat Detected::High risk eval found | Evaluates raw input\n"
            "::warning file=src/low.py,line=1,title=GPT Scan 30%25 Threat Detected::Low risk pattern\n"
        )
        parsed = gptscan.parse_github_annotations_content(raw_annotations)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["path"], "src/danger.py")
        self.assertEqual(parsed[0]["line"], "12")
        self.assertEqual(parsed[0]["own_conf"], "85%")
        self.assertEqual(parsed[0]["admin_desc"], "High risk eval found")
        self.assertEqual(parsed[0]["end-user_desc"], "Evaluates raw input")

        self.assertEqual(parsed[1]["path"], "src/low.py")
        self.assertEqual(parsed[1]["line"], "1")
        self.assertEqual(parsed[1]["own_conf"], "30%")
        self.assertEqual(parsed[1]["admin_desc"], "Low risk pattern")
        self.assertEqual(parsed[1]["end-user_desc"], "")

    def test_parse_report_content_auto_detect_annotations(self):
        content = "::error file=test.py,line=10,title=Threat 75%25::Suspicious code found"
        results = gptscan.parse_report_content(content)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "test.py")
        self.assertEqual(results[0]["line"], "10")
        self.assertEqual(results[0]["own_conf"], "75%")

    def test_parse_report_content_with_annotations_extension_hint(self):
        content = "::warning file=script.sh,line=3,title=Threat 40%25::Warning message"
        results = gptscan.parse_report_content(content, filename_hint="report.annotations")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "script.sh")
        self.assertEqual(results[0]["line"], "3")

    def test_export_results_to_file_github_format(self):
        results = [
            {"path": "a.py", "line": "1", "own_conf": "80%", "admin_desc": "Threat"}
        ]
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".annotations") as tf:
            tf_path = tf.name

        try:
            gptscan.export_results_to_file(tf_path, results, output_format="github")
            with open(tf_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("::error file=a.py,line=1,title=GPT Scan 80%25 Threat Detected::Threat", content)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_copy_as_github_annotations_gui(self):
        mock_tree = MagicMock()
        mock_tree.selection.return_value = ["item1"]

        test_item = ["a.py", "80%", "Admin desc", "User desc", "", "print(1)", "5"]

        with patch.object(gptscan, 'tree', mock_tree), \
             patch.object(gptscan, '_get_tree_results_as_dicts', return_value=[{
                 "path": "a.py", "line": "5", "own_conf": "80%", "admin_desc": "Admin desc", "end-user_desc": "User desc", "snippet": "print(1)"
             }]), \
             patch.object(gptscan, 'update_status') as mock_status:

            gptscan.copy_as_github_annotations()
            mock_tree.clipboard_clear.assert_called_once()
            mock_tree.clipboard_append.assert_called_once()
            appended = mock_tree.clipboard_append.call_args[0][0]
            self.assertIn("::error file=a.py,line=5,title=GPT Scan 80%25 Threat Detected::Admin desc | User desc", appended)
            mock_status.assert_called_once_with("Copied 1 item(s) as GitHub Annotations.")


if __name__ == "__main__":
    unittest.main()
