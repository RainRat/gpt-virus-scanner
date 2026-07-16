import os
import pytest
from pathlib import Path
from gptscan import analyze_filename, Config

# --- RTLO Detection ---

def test_analyze_filename_rtlo():
    rtlo_name = "invoice\u202egepj.exe"
    score, msg = analyze_filename(rtlo_name)
    assert score == 1.0
    assert "RTLO" in msg

# --- Double Extension Detection ---

def test_analyze_filename_double_extension_basic():
    score, msg = analyze_filename("test.pdf.exe")
    assert score == 0.9
    assert "double extension" in msg.lower()

    score, msg = analyze_filename("image.png.ps1")
    assert score == 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_double_extension_whitespace():
    # Space between extensions
    score, msg = analyze_filename("document.pdf .exe")
    assert score >= 0.9
    assert "double extension" in msg.lower()

    # Tab between extensions
    score, msg = analyze_filename("image.png\t.ps1")
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_double_extension_multiple_dots():
    score, msg = analyze_filename("invoice.pdf....exe")
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_double_extension_deceptive_prefixes():
    # .svg, .html, and .htm as deceptive prefixes
    for ext in [".svg", ".html", ".htm"]:
        name = f"page{ext}.exe"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_double_extension_exec_extensions():
    # .lnk and .inf as exec extensions
    for ext in [".lnk", ".inf"]:
        name = f"document.pdf{ext}"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_double_extension_benign():
    score, msg = analyze_filename("archive.tar.gz")
    assert score == 0.0

# --- Whitespace Tricks ---

def test_analyze_filename_large_whitespace_gap():
    name = "document.pdf          .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace" in msg.lower()

def test_analyze_filename_whitespace_before_extension():
    name = "malware .exe"
    score, msg = analyze_filename(name)
    assert score >= 0.5
    assert "whitespace" in msg.lower() or "space" in msg.lower()

def test_analyze_filename_normal_whitespace():
    score, msg = analyze_filename("my script.py")
    assert score == 0.0

# --- Trailing Characters ---

def test_analyze_filename_trailing_space():
    score, msg = analyze_filename("script.py ")
    assert score == 0.5
    assert "ends with a suspicious space" in msg

def test_analyze_filename_trailing_dots():
    score, msg = analyze_filename("file.txt.")
    assert score == 0.5
    assert "ends with a suspicious space or dot" in msg

    # Multiple trailing dots
    score, msg = analyze_filename("report.pdf...")
    assert score >= 0.5
    assert "dot" in msg.lower()

# --- Invisible and Control Characters ---

def test_analyze_filename_control_chars_bell():
    name = f"evil{chr(7)}.py"
    score, msg = analyze_filename(name)
    assert score == 0.7
    assert "control characters" in msg.lower()

def test_analyze_filename_unicode_invisible_chars():
    # Zero-width space (U+200B)
    score, msg = analyze_filename("file\u200b.exe")
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

    # Soft hyphen (U+00AD)
    score, msg = analyze_filename("invo\u00adice.exe")
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

# --- Integration with Config ---

def test_is_supported_file_with_suspicious_name():
    Config.initialize()
    Config.extensions_set = {".py"}

    assert Config.is_supported_file("invoice\u202egepj.exe") is True
    assert Config.is_supported_file("photo.jpg.exe") is True
    assert Config.is_supported_file("photo.jpg") is False
