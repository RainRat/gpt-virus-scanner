import pytest
from gptscan import analyze_filename

def test_analyze_filename_tab_gap():
    # Filename with large gap using tabs instead of spaces
    name = "invoice.pdf\t\t\t\t\t.exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace gap" in msg

def test_analyze_filename_mixed_gap():
    # Filename with mixed spaces and tabs
    name = "invoice.pdf  \t  \t  .exe"
    score, msg = analyze_filename(name)
    assert score == 0.9
    assert "whitespace gap" in msg

def test_analyze_filename_trailing_tab():
    name = "script.py\t"
    score, msg = analyze_filename(name)
    assert score == 0.5
    assert "suspicious space" in msg

def test_analyze_filename_empty_input():
    # os.path.basename('') is ''
    score, msg = analyze_filename("")
    assert score == 0.0
    assert msg == ""

def test_analyze_filename_multiple_dots_no_gap():
    # Not deceptive, just multiple dots
    name = "my.file.name.txt"
    score, msg = analyze_filename(name)
    assert score == 0.0

def test_analyze_filename_large_space_no_double_ext():
    # Large space but not trying to hide an extension
    name = "some file          with spaces.txt"
    score, msg = analyze_filename(name)
    assert score == 0.0
