import os
import pytest
from pathlib import Path
from gptscan import analyze_filename, Config

def test_analyze_filename_rtlo():
    # RTLO character U+202E
    rtlo_name = "invoice\u202egepj.exe"
    score, msg = analyze_filename(rtlo_name)
    assert score == 1.0
    assert "RTLO" in msg

def test_analyze_filename_double_extension():
    score, msg = analyze_filename("test.pdf.exe")
    assert score == 0.9
    assert "double extension" in msg

    score, msg = analyze_filename("image.png.ps1")
    assert score == 0.9
    assert "double extension" in msg

    # Benign double extension should not be flagged (or at least not with 0.9)
    score, msg = analyze_filename("archive.tar.gz")
    assert score == 0.0

def test_analyze_filename_hidden_whitespace():
    # Deceptive whitespace
    name = "document.pdf          .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace" in msg

    # Normal whitespace
    score, msg = analyze_filename("my script.py")
    assert score == 0.0

def test_analyze_filename_trailing():
    score, msg = analyze_filename("script.py ")
    assert score == 0.5
    assert "ends with a suspicious space" in msg

    score, msg = analyze_filename("file.txt.")
    assert score == 0.5
    assert "ends with a suspicious space or dot" in msg

def test_analyze_filename_control_chars():
    # ASCII 7 is Bell
    name = f"evil{chr(7)}.py"
    score, msg = analyze_filename(name)
    assert score == 0.7
    assert "control characters" in msg

def test_is_supported_file_with_suspicious_name():
    # Ensure Config is initialized
    Config.initialize()

    # Even if .exe is not in extensions_set, it should be supported if name is suspicious
    Config.extensions_set = {".py"}

    assert Config.is_supported_file("invoice\u202egepj.exe") is True
    assert Config.is_supported_file("photo.jpg.exe") is True

    # Benign non-supported should still be False
    assert Config.is_supported_file("photo.jpg") is False
