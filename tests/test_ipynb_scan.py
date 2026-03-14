import json
import pytest
from pathlib import Path
from gptscan import scan_files, Config

def test_ipynb_parsing(tmp_path):
    # Create a dummy Jupyter Notebook
    notebook_path = tmp_path / "test.ipynb"
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["import os\n", "os.system('rm -rf /')"]
            },
            {
                "cell_type": "markdown",
                "source": ["# This is a markdown cell"]
            },
            {
                "cell_type": "code",
                "source": "print('hello world')"
            }
        ]
    }
    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f)

    # Mock Config.is_supported_file to return True for our notebook
    # (Actually it should already work since we added .ipynb to defaults)

    events = list(scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True
    ))

    # Look for results in events
    results = [data for event_type, data in events if event_type == 'result']

    # We expect 2 results from the 2 code cells
    assert len(results) == 2
    assert any("test.ipynb [Cell 1]" in r[0] for r in results)
    assert any("test.ipynb [Cell 2]" in r[0] for r in results)
    # In dry_run, r[5] contains "(Snippet would be scanned, size: ...)"
    assert any("(Snippet would be scanned" in r[5] for r in results)

def test_invalid_ipynb_handled(tmp_path):
    # Create an invalid JSON file with .ipynb extension
    notebook_path = tmp_path / "invalid.ipynb"
    with open(notebook_path, "w") as f:
        f.write("{invalid json")

    events = list(scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True
    ))

    # The file should be skipped by the ipynb parser but might be picked up
    # as a regular file if extensions.txt says it's supported.
    # However, our logic in scan_files for .ipynb should catch the exception.
    results = [data for event_type, data in events if event_type == 'result']

    # Since it's invalid, it should NOT have [Cell X] results.
    # It might still be scanned as raw text if it falls through to the 'else' branch,
    # but my code does `non_archive_files.append(f_path)` in the Exception block.
    # Wait, my code does:
    # except Exception:
    #     non_archive_files.append(f_path)
    # So it will be scanned as raw text if parsing fails. This is acceptable fallback.

    assert not any("[Cell" in r[0] for r in results)
