import pytest
from pathlib import Path
from gptscan import Config, scan_files
import os

def test_is_supported_file_extension():
    # Save original
    original_exts = Config.extensions_set
    Config.extensions_set = {'.py', '.js'}
    try:
        assert Config.is_supported_file(Path("test.py")) is True
        assert Config.is_supported_file(Path("test.js")) is True
        assert Config.is_supported_file(Path("test.txt")) is False
    finally:
        Config.extensions_set = original_exts

def test_is_supported_file_shebang(tmp_path):
    original_exts = Config.extensions_set
    Config.extensions_set = {'.py'}
    try:
        # File with shebang but no extension
        f1 = tmp_path / "myscript"
        f1.write_bytes(b"#!/usr/bin/python\nprint('hello')")
        assert Config.is_supported_file(f1) is True

        # File with shebang and unknown extension
        f2 = tmp_path / "myscript.xyz"
        f2.write_bytes(b"#!/usr/bin/env node\nconsole.log('hi')")
        assert Config.is_supported_file(f2) is True

        # File with shebang for unsupported interpreter
        f3 = tmp_path / "other"
        f3.write_bytes(b"#!/bin/custom\n...")
        assert Config.is_supported_file(f3) is False

        # File without shebang and unknown extension
        f4 = tmp_path / "not_a_script"
        f4.write_bytes(b"some random content")
        assert Config.is_supported_file(f4) is False

        # Empty file
        f5 = tmp_path / "empty"
        f5.touch()
        assert Config.is_supported_file(f5) is False

        # Shell shebangs
        f6 = tmp_path / "sh_script"
        f6.write_bytes(b"#!/bin/sh\nls")
        assert Config.is_supported_file(f6) is True

        f7 = tmp_path / "bash_script"
        f7.write_bytes(b"#!/bin/bash\nls")
        assert Config.is_supported_file(f7) is True

    finally:
        Config.extensions_set = original_exts

def test_is_supported_file_explicit():
    assert Config.is_supported_file(Path("anything.txt"), is_explicit=True) is True

def test_scan_files_detection_logic(tmp_path, mocker):
    # Mock model and tensorflow to avoid loading them
    mocker.patch('gptscan.get_model')
    mocker.patch('gptscan._tf_module')

    # Setup files
    script_no_ext = tmp_path / "shebang_script"
    script_no_ext.write_bytes(b"#!/bin/bash\necho test")

    normal_py = tmp_path / "script.py"
    normal_py.touch()

    ignored_txt = tmp_path / "notes.txt"
    ignored_txt.touch()

    explicit_txt = tmp_path / "explicit.txt"
    explicit_txt.touch()

    original_exts = Config.extensions_set
    Config.extensions_set = {'.py'}

    try:
        results = []
        # Using dry_run=True to skip actual scanning logic
        # We pass both the directory and an explicit file
        for event_type, data in scan_files(
            [str(tmp_path), str(explicit_txt)],
            deep_scan=False,
            show_all=True,
            use_gpt=False,
            dry_run=True
        ):
            if event_type == 'result':
                results.append(data[0])

        assert str(script_no_ext) in results
        assert str(normal_py) in results
        assert str(explicit_txt) in results
        assert str(ignored_txt) not in results
    finally:
        Config.extensions_set = original_exts
