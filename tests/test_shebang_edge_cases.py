import pytest
from pathlib import Path
from gptscan import Config

def test_is_supported_file_shebang_windows_line_endings(tmp_path):
    f = tmp_path / "script_win"
    f.write_bytes(b"#!/usr/bin/env python\r\nprint('hello')")
    assert Config.is_supported_file(f) is True

def test_is_supported_file_shebang_env_split_parameters(tmp_path):
    f = tmp_path / "script_split"
    f.write_bytes(b"#!/usr/bin/env -S python3 -u\nprint('hello')")
    assert Config.is_supported_file(f) is True

def test_is_supported_file_shebang_mixed_case_and_multiple_spaces(tmp_path):
    f = tmp_path / "script_mixed"
    f.write_bytes(b"#!/usr/bin/Env\t  PyThOn3\nprint('hello')")
    assert Config.is_supported_file(f) is True

def test_is_supported_file_shebang_invalid_utf8_bytes(tmp_path):
    f = tmp_path / "script_bad_utf8"
    f.write_bytes(b"#!/usr/bin/env python\xff\xfe\nprint('hello')")
    assert Config.is_supported_file(f) is True

def test_is_supported_file_shebang_minimal_single_line_no_newline(tmp_path):
    f = tmp_path / "script_minimal"
    f.write_bytes(b"#!/usr/bin/env bash")
    assert Config.is_supported_file(f) is True

def test_is_supported_file_shebang_word_boundary_constraints(tmp_path):
    f = tmp_path / "script_boundary"
    f.write_bytes(b"#!/usr/bin/pythonabc\nprint('hello')")
    assert Config.is_supported_file(f) is False

    f2 = tmp_path / "script_boundary_2"
    f2.write_bytes(b"#!/usr/bin/bash-helper\nprint('hello')")
    assert Config.is_supported_file(f2) is False

def test_is_supported_file_shebang_extremely_long_shebang_line(tmp_path):
    f = tmp_path / "script_long"
    f.write_bytes(b"#!/usr/bin/env " + b" " * 150 + b"python\nprint('hello')")
    assert Config.is_supported_file(f) is False
