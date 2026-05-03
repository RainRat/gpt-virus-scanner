import pytest
from pathlib import Path
import gptscan

def test_scan_summary_excludes_skipped_large_files(tmp_path, monkeypatch):
    """Verify that files skipped due to size limits are not counted in the final summary."""
    target_dir = tmp_path / "test_summary"
    target_dir.mkdir()

    # Create a small file (should be scanned)
    small_file = target_dir / "small.py"
    small_file.write_text("print('small')")
    small_size = small_file.stat().st_size

    # Create a large file (should be skipped)
    large_file = target_dir / "large.py"
    large_file.write_text("print('large')" * 1000)

    # Set limit to be between small and large file sizes
    monkeypatch.setattr(gptscan.Config, 'MAX_FILE_SIZE', 100)

    # Scan the directory (so files are not explicit)
    # We must mock get_model and _tf_module to avoid loading the actual model
    from unittest.mock import MagicMock
    monkeypatch.setattr(gptscan, 'get_model', lambda: MagicMock())
    monkeypatch.setattr(gptscan, '_tf_module', MagicMock())

    gen = gptscan.scan_files([str(target_dir)], deep_scan=False, show_all=True, use_gpt=False)

    summary = None
    results = []
    for event_type, data in gen:
        if event_type == 'result':
            results.append(data)
        elif event_type == 'summary':
            summary = data

    # results[0] should be the Large File result indicating it was skipped
    # results[1] should be the small.py result

    assert any(r[0] == str(large_file) and r[1] == 'Large File' for r in results)
    assert any(r[0] == str(small_file) for r in results)

    assert summary is not None
    total_files, total_bytes, elapsed_time = summary

    # ONLY small_file should be counted
    assert total_files == 1
    assert total_bytes == small_size
