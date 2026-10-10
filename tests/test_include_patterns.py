import pytest
from pathlib import Path
from unittest.mock import MagicMock
import gptscan


def test_scan_files_include_patterns_filter(tmp_path, monkeypatch):
    # Create test files
    py_file = tmp_path / "script.py"
    py_file.write_text("print('hello')", encoding="utf-8")

    js_file = tmp_path / "app.js"
    js_file.write_text("console.log('hello');", encoding="utf-8")

    txt_file = tmp_path / "notes.py"  # Use .py extension so it passes is_supported_file
    txt_file.write_text("print('some notes')", encoding="utf-8")

    sub_dir = tmp_path / "src"
    sub_dir.mkdir()
    sub_py = sub_dir / "module.py"
    sub_py.write_text("import os", encoding="utf-8")

    # Mock get_model
    monkeypatch.setattr(gptscan, "get_model", lambda: None)

    # 1. Include only *.py files
    events = list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, dry_run=True, include_patterns=["*.py"]))
    scanned_paths = [Path(args[0]) for evt, args in events if evt == "result"]
    assert py_file in scanned_paths
    assert sub_py in scanned_paths
    assert txt_file in scanned_paths
    assert js_file not in scanned_paths

    # 2. Include multiple patterns (*.js, notes.py)
    events_multi = list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, dry_run=True, include_patterns=["*.js", "notes.py"]))
    scanned_paths_multi = [Path(args[0]) for evt, args in events_multi if evt == "result"]
    assert js_file in scanned_paths_multi
    assert txt_file in scanned_paths_multi
    assert py_file not in scanned_paths_multi
    assert sub_py not in scanned_paths_multi

    # 3. Parent path matching (src/*)
    events_parent = list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, dry_run=True, include_patterns=["src/*"]))
    scanned_paths_parent = [Path(args[0]) for evt, args in events_parent if evt == "result"]
    assert sub_py in scanned_paths_parent
    assert py_file not in scanned_paths_parent
    assert js_file not in scanned_paths_parent


def test_scan_files_include_and_exclude_interaction(tmp_path, monkeypatch):
    py1 = tmp_path / "test1.py"
    py1.write_text("print(1)", encoding="utf-8")
    py2 = tmp_path / "test2.py"
    py2.write_text("print(2)", encoding="utf-8")
    js1 = tmp_path / "test1.js"
    js1.write_text("console.log(1)", encoding="utf-8")

    monkeypatch.setattr(gptscan, "get_model", lambda: None)

    # Include *.py but exclude test2.py
    events = list(gptscan.scan_files(
        str(tmp_path),
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True,
        include_patterns=["*.py"],
        exclude_patterns=["test2.py"]
    ))
    scanned_paths = [Path(args[0]) for evt, args in events if evt == "result"]
    assert py1 in scanned_paths
    assert py2 not in scanned_paths
    assert js1 not in scanned_paths


def test_cli_include_flag(tmp_path, monkeypatch, capsys):
    f1 = tmp_path / "a.py"
    f1.write_text("import os\nos.system('ls')", encoding="utf-8")
    f2 = tmp_path / "b.js"
    f2.write_text("eval('alert(1)')", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.9]]
    mock_tf = MagicMock()
    mock_tf.constant.side_effect = lambda x: x
    mock_tf.expand_dims.side_effect = lambda x, axis: x

    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    ret = gptscan.run_cli(
        targets=str(tmp_path),
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        dry_run=False,
        include_patterns=["*.py"],
        quiet=True,
        paths_only=True
    )
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]

    assert str(f1) in lines or any("a.py" in l for l in lines)
    assert not any("b.js" in l for l in lines)
