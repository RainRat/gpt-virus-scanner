import pytest
from gptscan import analyze_filename

def test_analyze_filename_hidden_whitespace_tabs():
    name = "invoice.pdf\t\t\t\t\t.exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace gap" in msg

def test_analyze_filename_hidden_whitespace_mixed():
    name = "invoice.pdf  \t  \t  .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace gap" in msg

def test_analyze_filename_hidden_whitespace_no_dot_prefix():
    name = "malicious     .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace gap" in msg

def test_analyze_filename_trailing_tab():
    name = "script.py\t"
    score, msg = analyze_filename(name)
    assert score == 0.5
    assert "whitespace or dot" in msg

def test_analyze_filename_trailing_newline():
    name = "script.py\n"
    score, msg = analyze_filename(name)
    assert score == 0.5
    assert "whitespace or dot" in msg

def test_analyze_filename_large_gap_non_exec():
    name = "readme     .txt"
    score, msg = analyze_filename(name)
    assert score == 0.0

def test_analyze_filename_empty():
    score, msg = analyze_filename("")
    assert score == 0.0
    assert msg == ""
