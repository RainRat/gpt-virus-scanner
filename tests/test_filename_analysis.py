import os
import pytest
from pathlib import Path
from gptscan import analyze_filename, Config

def test_analyze_filename_rtlo():
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

    score, msg = analyze_filename("archive.tar.gz")
    assert score == 0.0

def test_analyze_filename_hidden_whitespace():
    name = "document.pdf          .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace" in msg

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
    name = f"evil{chr(7)}.py"
    score, msg = analyze_filename(name)
    assert score == 0.7
    assert "control characters" in msg

def test_is_supported_file_with_suspicious_name():
    Config.initialize()
    Config.extensions_set = {".py"}

    assert Config.is_supported_file("invoice\u202egepj.exe") is True
    assert Config.is_supported_file("photo.jpg.exe") is True
    assert Config.is_supported_file("photo.jpg") is False

def test_analyze_filename_whitespace_double_extension():
    name = "document.pdf .exe"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_tab_double_extension():
    name = "image.png\t.ps1"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_multiple_dots_double_extension():
    name = "invoice.pdf....exe"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_new_deceptive_prefixes():
    for ext in [".svg", ".html", ".htm"]:
        name = f"page{ext}.exe"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_new_exec_extensions():
    for ext in [".lnk", ".inf"]:
        name = f"document.pdf{ext}"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_whitespace_before_single_extension():
    name = "malware .exe"
    score, msg = analyze_filename(name)
    assert score >= 0.5
    assert "whitespace" in msg.lower() or "space" in msg.lower()

def test_analyze_filename_unicode_invisible_chars():
    name = "file\u200b.exe"
    score, msg = analyze_filename(name)
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

    name = "invo\u00adice.exe"
    score, msg = analyze_filename(name)
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

def test_analyze_filename_trailing_dots_complex():
    name = "report.pdf..."
    score, msg = analyze_filename(name)
    assert score >= 0.5
    assert "dot" in msg.lower()
