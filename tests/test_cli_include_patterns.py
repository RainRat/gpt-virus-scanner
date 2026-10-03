import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

import gptscan


def test_scan_files_with_include_patterns(tmp_path):
    # Create test directory structure
    py_file = tmp_path / "app.py"
    py_file.write_text("print('hello')", encoding="utf-8")

    js_file = tmp_path / "script.js"
    js_file.write_text("console.log('hello');", encoding="utf-8")

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("some text", encoding="utf-8")

    sub_dir = tmp_path / "src"
    sub_dir.mkdir()
    sub_py = sub_dir / "lib.py"
    sub_py.write_text("def fn(): pass", encoding="utf-8")

    # 1. Include only *.py
    gen = gptscan.scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True,
        include_patterns=["*.py"]
    )
    results = [data[0] for event_type, data in gen if event_type == "result"]
    assert any("app.py" in r for r in results)
    assert any("lib.py" in r for r in results)
    assert not any("script.js" in r for r in results)
    assert not any("notes.txt" in r for r in results)

    # 2. Include subfolder pattern src/*
    gen_src = gptscan.scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True,
        include_patterns=["src/*"]
    )
    results_src = [data[0] for event_type, data in gen_src if event_type == "result"]
    assert len(results_src) == 1
    assert "lib.py" in results_src[0]


def test_scan_files_include_and_exclude_combined(tmp_path):
    py_file = tmp_path / "app.py"
    py_file.write_text("print('app')", encoding="utf-8")

    test_py = tmp_path / "test_app.py"
    test_py.write_text("print('test')", encoding="utf-8")

    js_file = tmp_path / "app.js"
    js_file.write_text("console.log('js')", encoding="utf-8")

    gen = gptscan.scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        dry_run=True,
        include_patterns=["*.py"],
        exclude_patterns=["test_*"]
    )
    results = [data[0] for event_type, data in gen if event_type == "result"]
    assert any("app.py" in r for r in results)
    assert not any("test_app.py" in r for r in results)
    assert not any("app.js" in r for r in results)


def test_cli_include_option(tmp_path, monkeypatch, capsys):
    py_file = tmp_path / "main.py"
    py_file.write_text("import os", encoding="utf-8")

    sh_file = tmp_path / "script.sh"
    sh_file.write_text("#!/bin/bash\necho 1", encoding="utf-8")

    test_args = ["gptscan.py", str(tmp_path), "--cli", "-i", "*.py", "--show-all"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    captured = capsys.readouterr()
    assert "main.py" in captured.out or "main.py" in captured.err


def test_cli_include_file_option(tmp_path, monkeypatch, capsys):
    py_file = tmp_path / "service.py"
    py_file.write_text("import sys", encoding="utf-8")

    inc_file = tmp_path / ".gptscaninclude"
    inc_file.write_text("# Comment line\n\n*.py # inline comment\n", encoding="utf-8")

    test_args = ["gptscan.py", str(tmp_path), "--cli", "--include-file", str(inc_file), "--show-all"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    captured = capsys.readouterr()
    assert "service.py" in captured.out or "service.py" in captured.err


def test_cli_include_file_unreadable(tmp_path, monkeypatch):
    missing_file = tmp_path / "nonexistent_include.txt"
    test_args = ["gptscan.py", str(tmp_path), "--cli", "--include-file", str(missing_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc:
        gptscan.main()
    assert exc.value.code != 0
