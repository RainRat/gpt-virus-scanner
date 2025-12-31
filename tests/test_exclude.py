import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import gptscan

def test_scan_files_exclude_patterns(tmp_path):
    # Setup directories and files
    keep_dir = tmp_path / "keep"
    ignore_dir = tmp_path / "ignore"
    keep_dir.mkdir()
    ignore_dir.mkdir()

    (keep_dir / "good.py").write_text("print('good')")
    (ignore_dir / "bad.py").write_text("print('bad')")
    (tmp_path / "root.py").write_text("print('root')")

    # Mock collect_files to avoid depending on file system traversal implementation details,
    # but since scan_files calls collect_files, we can just let it run if we point it to tmp_path.
    # However, collect_files uses Path.rglob('*'), which works on tmp_path.

    # We need to mock Config.extensions_set to include .py
    with patch.object(gptscan.Config, 'extensions_set', {'.py'}):
        # Case 1: Exclude "ignore/*"
        scan_gen = gptscan.scan_files(
            scan_targets=[str(tmp_path)],
            deep_scan=False,
            show_all=True, # Show all so we see results even if low confidence
            use_gpt=False,
            dry_run=True,
            exclude_patterns=["ignore/*"]
        )

        results = []
        for event, data in scan_gen:
            if event == 'result':
                results.append(data[0]) # path

        # We expect root.py and keep/good.py. ignore/bad.py should be missing.
        # Note: paths returned are strings.
        assert any("good.py" in r for r in results)
        assert any("root.py" in r for r in results)
        assert not any("bad.py" in r for r in results)

def test_scan_files_exclude_multiple_patterns(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.js").write_text("")
    (tmp_path / "c.txt").write_text("")

    with patch.object(gptscan.Config, 'extensions_set', {'.py', '.js', '.txt'}):
        scan_gen = gptscan.scan_files(
            scan_targets=[str(tmp_path)],
            deep_scan=False,
            show_all=True,
            use_gpt=False,
            dry_run=True,
            exclude_patterns=["*.js", "*.txt"]
        )

        results = [data[0] for event, data in scan_gen if event == 'result']

        assert any("a.py" in r for r in results)
        assert not any("b.js" in r for r in results)
        assert not any("c.txt" in r for r in results)
