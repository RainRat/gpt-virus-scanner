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

        # PowerShell shebangs
        f8 = tmp_path / "pwsh_script"
        f8.write_bytes(b"#!/usr/bin/pwsh\nGet-ChildItem")
        assert Config.is_supported_file(f8) is True

        f9 = tmp_path / "powershell_script"
        f9.write_bytes(b"#!/usr/bin/env powershell\nls")
        assert Config.is_supported_file(f9) is True

        # Regression test for substring false positives (e.g., 'sh' in 'ships')
        f10 = tmp_path / "ships_script"
        f10.write_bytes(b"#!/usr/bin/ships\n")
        assert Config.is_supported_file(f10) is False

        f11 = tmp_path / "mysh_script"
        f11.write_bytes(b"#!/usr/bin/mysh\n")
        assert Config.is_supported_file(f11) is False

        # New interpreters
        f12 = tmp_path / "lua_script"
        f12.write_bytes(b"#!/usr/bin/lua\n")
        assert Config.is_supported_file(f12) is True

        f13 = tmp_path / "nodejs_script"
        f13.write_bytes(b"#!/usr/bin/nodejs\n")
        assert Config.is_supported_file(f13) is True

        f14 = tmp_path / "ipython_script"
        f14.write_bytes(b"#!/usr/bin/ipython\n")
        assert Config.is_supported_file(f14) is True

        f15 = tmp_path / "ash_script"
        f15.write_bytes(b"#!/bin/ash\n")
        assert Config.is_supported_file(f15) is True

        f16 = tmp_path / "dash_script"
        f16.write_bytes(b"#!/bin/dash\n")
        assert Config.is_supported_file(f16) is True

        f17 = tmp_path / "osascript_script"
        f17.write_bytes(b"#!/usr/bin/osascript\n")
        assert Config.is_supported_file(f17) is True

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
