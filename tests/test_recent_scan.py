import os
import time
import pytest
from pathlib import Path
from gptscan import parse_duration, collect_files

def test_parse_duration():
    assert parse_duration("1s") == 1.0
    assert parse_duration("1m") == 60.0
    assert parse_duration("1h") == 3600.0
    assert parse_duration("1d") == 86400.0
    assert parse_duration("1w") == 604800.0
    assert parse_duration("24h") == 24 * 3600.0
    assert parse_duration("1.5h") == 1.5 * 3600.0
    assert parse_duration(" 1h ") == 3600.0
    assert parse_duration("1") == 3600.0  # Default unit is hours
    assert parse_duration("") is None
    assert parse_duration("invalid") is None
    assert parse_duration("1x") is None

def test_collect_files_modified_since(tmp_path):
    # Create test files
    old_file = tmp_path / "old.py"
    new_file = tmp_path / "new.py"

    old_file.write_text("print('old')")
    new_file.write_text("print('new')")

    # Set modification times
    now = time.time()
    os.utime(old_file, (now - 10000, now - 10000))  # ~2.7 hours ago
    os.utime(new_file, (now, now))

    # Test filtering
    # 1. No filter
    files = collect_files([str(tmp_path)])
    assert len(files) == 2

    # 2. Filter files modified in the last 1 hour (3600s)
    since = now - 3600
    files = collect_files([str(tmp_path)], modified_since=since)
    assert len(files) == 1
    assert Path(files[0]).name == "new.py"

    # 3. Filter files modified in the last 4 hours (14400s)
    since = now - 14400
    files = collect_files([str(tmp_path)], modified_since=since)
    assert len(files) == 2

def test_collect_files_explicit_target_modified_since(tmp_path):
    # Even if explicitly passed, it should be filtered if it doesn't match
    old_file = tmp_path / "old.py"
    old_file.write_text("print('old')")

    now = time.time()
    os.utime(old_file, (now - 10000, now - 10000))

    since = now - 3600
    files = collect_files([str(old_file)], modified_since=since)
    assert len(files) == 0
