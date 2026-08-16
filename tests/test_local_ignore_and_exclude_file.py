import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import gptscan

def test_parse_ignore_file_basic(tmp_path):
    """Test that parse_ignore_file correctly extracts non-comment, non-empty pattern lines."""
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text(
        "# This is a comment\n"
        "node_modules/ # ignore packages\n"
        "\n"
        "venv/\n"
        "  *.log  # ignore logs\n"
        "  # nested comment\n"
    )

    patterns = gptscan.parse_ignore_file(ignore_file)
    assert patterns == ["node_modules/", "venv/", "*.log"]


def test_discover_local_ignore_patterns(tmp_path):
    """Test recursive discovery of local ignore files in target folders."""
    # Setup subfolders
    sub1 = tmp_path / "sub1"
    sub2 = tmp_path / "sub2"
    sub1.mkdir()
    sub2.mkdir()

    # Create ignore files
    ignore1 = sub1 / ".gptscanignore"
    ignore1.write_text("*.tmp\n")

    ignore2 = sub2 / ".gitignore"
    ignore2.write_text("build/\n")

    # Call discover_local_ignore_patterns
    ignore_rules = gptscan.discover_local_ignore_patterns([str(tmp_path)])

    # We expect 2 rules
    assert len(ignore_rules) == 2

    # Verify rule details (each tuple is (dir_path, pattern))
    dirs = {rule[0].resolve(): rule[1] for rule in ignore_rules}
    assert dirs[sub1.resolve()] == "*.tmp"
    assert dirs[sub2.resolve()] == "build/"


def test_discover_local_ignore_patterns_file_target(tmp_path):
    """Test local ignore discovery when the target is a single file."""
    # Setup parent folder with ignore file
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text("*.tmp\n")

    test_file = tmp_path / "script.py"
    test_file.touch()

    # Discovering when targeting the file specifically
    ignore_rules = gptscan.discover_local_ignore_patterns([str(test_file)])

    assert len(ignore_rules) == 1
    assert ignore_rules[0][0].resolve() == tmp_path.resolve()
    assert ignore_rules[0][1] == "*.tmp"


def test_local_ignore_filtering_scan_files(tmp_path, monkeypatch):
    """Verify that scan_files excludes files matching discovered local ignore patterns recursively."""
    # Mock tensorflow and keras model to prevent actual load
    mock_model = MagicMock()
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    monkeypatch.setattr(gptscan, "_tf_module", MagicMock())

    # Create directory structure
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    src_dir = project_dir / "src"
    src_dir.mkdir()

    # Create files
    file_to_keep = src_dir / "main.py"
    file_to_keep.write_text("print('hello')")

    file_to_ignore_tmp = src_dir / "test.tmp"
    file_to_ignore_tmp.write_text("temporary data")

    file_to_ignore_log = project_dir / "app.log"
    file_to_ignore_log.write_text("error log")

    # Create local ignore files
    (project_dir / ".gptscanignore").write_text("*.log\n")
    (src_dir / ".gitignore").write_text("*.tmp\n")

    # Execute file scan
    # All of the collected files should be filtered
    event_gen = gptscan.scan_files(
        scan_targets=[str(project_dir)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True  # Dry run is safe and avoids any ML model prediction
    )

    events = list(event_gen)
    # Filter 'result' events which yield (path, ...)
    scanned_paths = [data[0] for ev, data in events if ev == 'result']

    # The file_to_keep should be scanned (or yielded)
    assert any("main.py" in str(p) for p in scanned_paths)

    # The ignored files should NOT be scanned/yielded
    assert not any("test.tmp" in str(p) for p in scanned_paths)
    assert not any("app.log" in str(p) for p in scanned_paths)


def test_exclude_file_argparse_cli(tmp_path, monkeypatch):
    """Test that `--exclude-file` works on CLI by parsing arguments correctly."""
    exclude_file = tmp_path / "custom_excludes.txt"
    exclude_file.write_text(
        "custom_pattern1\n"
        "# comment\n"
        "custom_pattern2 # trailing\n"
    )

    # Mock the parser and main block execution
    # Let's test the argparse logic specifically
    test_args = [
        "gptscan.py",
        "some_folder",
        "--cli",
        "--exclude-file", str(exclude_file)
    ]

    with patch("sys.argv", test_args):
        # Create an ArgumentParser and parse
        import argparse
        # We can extract the parser or call it
        # Let's verify our added `--exclude-file` argument parses correctly
        # Let's run parser directly
        from gptscan import main
        # We patch sys.exit to prevent actual CLI exit
        # We can also patch run_cli to check what final_excludes was passed to it
        mock_run_cli = MagicMock(return_value=0)
        monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

        try:
            main()
        except SystemExit:
            pass

        # Check if run_cli was called with the loaded exclusions
        args, kwargs = mock_run_cli.call_args
        exclude_patterns = kwargs.get("exclude_patterns", [])

        assert "custom_pattern1" in exclude_patterns
        assert "custom_pattern2" in exclude_patterns
        assert "# comment" not in exclude_patterns


def test_parse_ignore_file_unreadable_exception(tmp_path, monkeypatch):
    unreadable_file = tmp_path / "unreadable.ignore"
    unreadable_file.touch()

    def mock_open_raise(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("builtins.open", mock_open_raise)

    patterns = gptscan.parse_ignore_file(unreadable_file)
    assert patterns == []


def test_discover_local_ignore_patterns_rglob_exception(tmp_path, monkeypatch):
    target_dir = tmp_path / "scan_target"
    target_dir.mkdir()

    def mock_rglob_raise(*args, **kwargs):
        raise OSError("Directory read error")

    monkeypatch.setattr(Path, "rglob", mock_rglob_raise)

    ignore_rules = gptscan.discover_local_ignore_patterns([str(target_dir)])
    assert ignore_rules == []


def test_discover_local_ignore_patterns_nonexistent_and_invalid_targets():
    ignore_rules = gptscan.discover_local_ignore_patterns(["/nonexistent/path/for/test"])
    assert ignore_rules == []
