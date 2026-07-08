import pytest
from gptscan import analyze_filename

def test_analyze_filename_whitespace_double_extension():
    # Gap identified: space between extensions
    name = "document.pdf .exe"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_tab_double_extension():
    # Gap identified: tab between extensions
    name = "image.png\t.ps1"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_multiple_dots_double_extension():
    # Gap identified: multiple dots between extensions
    name = "invoice.pdf....exe"
    score, msg = analyze_filename(name)
    assert score >= 0.9
    assert "double extension" in msg.lower()

def test_analyze_filename_new_deceptive_prefixes():
    # Gap identified: .svg and .html as deceptive prefixes
    for ext in [".svg", ".html", ".htm"]:
        name = f"page{ext}.exe"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_new_exec_extensions():
    # Gap identified: .lnk and .inf as exec extensions
    for ext in [".lnk", ".inf"]:
        name = f"document.pdf{ext}"
        score, msg = analyze_filename(name)
        assert score >= 0.9
        assert "double extension" in msg.lower()

def test_analyze_filename_whitespace_before_single_extension():
    # Gap identified: whitespace before single extension dot
    name = "malware .exe"
    score, msg = analyze_filename(name)
    assert score >= 0.5
    assert "whitespace" in msg.lower() or "space" in msg.lower()

def test_analyze_filename_unicode_invisible_chars():
    # Gap identified: Zero-width space (U+200B)
    name = "file\u200b.exe"
    score, msg = analyze_filename(name)
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

    # Soft hyphen (U+00AD)
    name = "invo\u00adice.exe"
    score, msg = analyze_filename(name)
    assert score >= 0.7
    assert "invisible" in msg.lower() or "control" in msg.lower()

def test_analyze_filename_trailing_dots_complex():
    # Multiple trailing dots
    name = "report.pdf..."
    score, msg = analyze_filename(name)
    assert score >= 0.5
    assert "dot" in msg.lower()
