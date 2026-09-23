import pytest
from pathlib import Path
import sys
import subprocess
from gptscan import collect_files, run_cli


def test_collect_files_max_depth_levels(tmp_path):
    # Create directory structure:
    # root/
    # ├── top.py             (depth 1)
    # ├── sub1/
    # │   ├── file1.py       (depth 2)
    # │   └── sub2/
    # │       └── file2.py   (depth 3)
    top_file = tmp_path / "top.py"
    top_file.write_text("print('top')")

    sub1_dir = tmp_path / "sub1"
    sub1_dir.mkdir()
    file1 = sub1_dir / "file1.py"
    file1.write_text("print('file1')")

    sub2_dir = sub1_dir / "sub2"
    sub2_dir.mkdir()
    file2 = sub2_dir / "file2.py"
    file2.write_text("print('file2')")

    # max_depth=None -> all 3 files
    files_all = collect_files(str(tmp_path), max_depth=None)
    assert len(files_all) == 3

    # max_depth=1 -> top.py only
    files_d1 = collect_files(str(tmp_path), max_depth=1)
    assert len(files_d1) == 1
    assert files_d1[0].resolve() == top_file.resolve()

    # max_depth=2 -> top.py and file1.py
    files_d2 = collect_files(str(tmp_path), max_depth=2)
    assert len(files_d2) == 2
    paths_d2 = {f.resolve() for f in files_d2}
    assert paths_d2 == {top_file.resolve(), file1.resolve()}

    # max_depth=0 -> no files under directory
    files_d0 = collect_files(str(tmp_path), max_depth=0)
    assert len(files_d0) == 0


def test_collect_files_max_depth_explicit_file(tmp_path):
    sub_dir = tmp_path / "a" / "b" / "c"
    sub_dir.mkdir(parents=True)
    deep_file = sub_dir / "deep.py"
    deep_file.write_text("print('deep')")

    # Explicit file target is returned directly
    collected = collect_files(str(deep_file), max_depth=1)
    assert len(collected) == 1
    assert collected[0].resolve() == deep_file.resolve()


def test_cli_max_depth_integration(tmp_path, capsys):
    top_file = tmp_path / "top.py"
    top_file.write_text("import os\nos.system('calc')")

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    deep_file = sub_dir / "deep.py"
    deep_file.write_text("import os\nos.system('calc')")

    # run_cli with max_depth=1 should only scan top.py
    count = run_cli(
        targets=str(tmp_path),
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        dry_run=True,
        max_depth=1
    )
    captured = capsys.readouterr()
    assert "top.py" in captured.out
    assert "deep.py" not in captured.out


def test_cli_invalid_max_depth():
    cmd = [sys.executable, "gptscan.py", "--max-depth", "-1", "--cli"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode != 0
    assert "Value for --max-depth must be a non-negative integer" in res.stderr


def test_collect_files_negative_depth_and_oserror(tmp_path, monkeypatch):
    test_file = tmp_path / "test.py"
    test_file.write_text("print(1)")

    # Negative depth should return no files from directory
    assert collect_files(str(tmp_path), max_depth=-1) == []

    # If rglob raises OSError during directory traversal, it should be handled gracefully
    def mock_rglob(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "rglob", mock_rglob)
    assert collect_files(str(tmp_path), max_depth=1) == []
    assert collect_files(str(tmp_path)) == []
