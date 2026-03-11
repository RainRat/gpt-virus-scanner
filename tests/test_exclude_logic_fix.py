
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_exclude_paths_consolidated_logic(tmp_path, monkeypatch):
    # Set up a dummy .gptscanignore
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text("existing.py\n")

    # Change CWD to tmp_path so relpath works predictably
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "exists", lambda p: str(p) == ".gptscanignore" or p == Path(".gptscanignore"))

    # Mock Config
    monkeypatch.setattr(gptscan.Config, "ignore_patterns", ["existing.py"])

    # Mock messagebox
    monkeypatch.setattr(gptscan.messagebox, "askyesno", lambda title, msg: True)

    # Mock _apply_filter and update_status
    monkeypatch.setattr(gptscan, "_apply_filter", MagicMock())
    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    # Initialize cache
    gptscan._all_results_cache = [
        ("file1.py", "10%", "", "", "", "print(1)", "1"),
        ("file2.py", "20%", "", "", "", "print(2)", "1"),
        ("existing.py", "30%", "", "", "", "print(3)", "1")
    ]

    # Test exclusion
    paths_to_exclude = ["file1.py", "file2.py"]

    # We need to mock open specifically for .gptscanignore in the current directory
    # But it's easier to just let it use the real filesystem in tmp_path since we chdir'd

    result = gptscan.exclude_paths(paths_to_exclude, confirm=False)

    assert result is True

    # Check .gptscanignore content
    content = ignore_file.read_text()
    assert "file1.py" in content
    assert "file2.py" in content
    assert content.count("file1.py") == 1

    # Check Config.ignore_patterns
    assert "file1.py" in gptscan.Config.ignore_patterns
    assert "file2.py" in gptscan.Config.ignore_patterns

    # Check _all_results_cache
    assert len(gptscan._all_results_cache) == 1
    assert gptscan._all_results_cache[0][0] == "existing.py"
