
import pytest
import os
from pathlib import Path
from gptscan import scan_files, Config

def test_total_bytes_scanned_skips_large_files(tmp_path):
    # Setup: Create a large file and a small file
    # Ensure Config.MAX_FILE_SIZE is set to a known small value for the test
    original_max_size = Config.MAX_FILE_SIZE
    Config.MAX_FILE_SIZE = 100  # 100 bytes

    try:
        large_file = tmp_path / "large.py"
        large_content = b"P" * 200
        large_file.write_bytes(large_content)

        small_file = tmp_path / "small.py"
        small_content = b"print('hello')"
        small_file.write_bytes(small_content)

        # Run scan_files
        # We need to pass targets as a list of strings
        gen = scan_files([str(tmp_path)], deep_scan=False, show_all=True, use_gpt=False)

        summary = None
        results = []
        for event_type, data in gen:
            if event_type == 'result':
                results.append(data)
            elif event_type == 'summary':
                summary = data

        assert summary is not None
        total_scanned, total_bytes, _ = summary

        # Verify that large.py was skipped
        large_results = [r for r in results if "large.py" in r[0]]
        assert len(large_results) > 0
        assert any("Large File" in str(r[1]) for r in large_results)

        # The bug: total_bytes currently includes large_content size
        # Correct behavior: total_bytes should only include small_content size
        print(f"Total bytes reported: {total_bytes}")
        print(f"Small file size: {len(small_content)}")
        print(f"Large file size: {len(large_content)}")

        assert total_bytes == len(small_content)

    finally:
        Config.MAX_FILE_SIZE = original_max_size
