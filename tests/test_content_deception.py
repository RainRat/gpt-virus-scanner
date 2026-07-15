import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import gptscan
from gptscan import Config, analyze_filename, scan_files

@pytest.fixture
def mock_scan_env(monkeypatch):
    """Mocks TensorFlow and Model for scan_files tests."""
    mock_predict = MagicMock(return_value=[[0.1]])
    mock_model = SimpleNamespace(predict=mock_predict)
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    mock_tf = SimpleNamespace(
        constant=lambda x: x,
        expand_dims=lambda x, axis: x,
    )
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)
    return mock_predict

def test_is_supported_file_content_deception(tmp_path):
    Config.initialize()
    Config.extensions_set = {".py", ".sh"}
    Config.scan_all_files = False

    # 1. Create a "JPG" that is actually a Windows Executable (MZ)
    deceptive_jpg = tmp_path / "normal_image.jpg"
    deceptive_jpg.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")

    # Now it SHOULD be True
    assert Config.is_supported_file(deceptive_jpg) is True

def test_is_supported_file_shebang_deception(tmp_path):
    Config.initialize()
    Config.extensions_set = {".py", ".sh"}
    Config.scan_all_files = False

    # 2. Create a "TXT" that is actually a Shell script
    deceptive_txt = tmp_path / "readme.txt"
    deceptive_txt.write_bytes(b"#!/bin/bash\necho 'hello'")

    # This was already True, should remain True
    assert Config.is_supported_file(deceptive_txt) is True

def test_is_supported_file_elf_deception(tmp_path):
    Config.initialize()
    Config.extensions_set = {".py", ".sh"}
    Config.scan_all_files = False

    # 3. Create a "PDF" that is actually an ELF executable
    deceptive_pdf = tmp_path / "document.pdf"
    deceptive_pdf.write_bytes(b"\x7fELF\x02\x01\x01\x00")

    # Now it SHOULD be True
    assert Config.is_supported_file(deceptive_pdf) is True

def test_analyze_content_mismatch_windows():
    from gptscan import analyze_content_mismatch
    score, msg = analyze_content_mismatch("test.jpg", b"MZ\x90\x00")
    assert score == 1.0
    assert "Windows executable" in msg
    assert ".jpg" in msg

def test_analyze_content_mismatch_linux():
    from gptscan import analyze_content_mismatch
    score, msg = analyze_content_mismatch("script.pdf", b"\x7fELF\x02")
    assert score == 1.0
    assert "Linux executable" in msg
    assert ".pdf" in msg

def test_analyze_content_mismatch_mac():
    from gptscan import analyze_content_mismatch
    # Mach-O 64-bit
    score, msg = analyze_content_mismatch("data.png", b"\xcf\xfa\xed\xfe")
    assert score == 1.0
    assert "macOS executable" in msg

def test_analyze_content_mismatch_shebang():
    from gptscan import analyze_content_mismatch
    score, msg = analyze_content_mismatch("notes.docx", b"#!/usr/bin/python\n")
    assert score == 0.8
    assert "Script shebang" in msg
    assert ".docx" in msg

def test_analyze_content_mismatch_benign():
    from gptscan import analyze_content_mismatch
    # JPG starting with its real magic
    score, msg = analyze_content_mismatch("image.jpg", b"\xff\xd8\xff")
    assert score == 0.0

    # .py file with MZ (we don't flag script extensions for binary mismatch here)
    score, msg = analyze_content_mismatch("test.py", b"MZ\x90\x00")
    assert score == 0.0

def test_scan_files_detects_deception(mock_scan_env, tmp_path):
    """End-to-end test verifying scan_files flags deceptive content."""
    Config.initialize()
    Config.extensions_set = {".py"}
    Config.THRESHOLD = 50

    deceptive_jpg = tmp_path / "evil.jpg"
    deceptive_jpg.write_bytes(b"MZ\x90\x00\x03\x00")

    cancel_event = SimpleNamespace(is_set=lambda: False)

    events = list(scan_files(
        str(tmp_path),
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        cancel_event=cancel_event
    ))

    # Filter for results
    results = [data for event_type, data in events if event_type == 'result']

    # Should find our deceptive file
    found = [r for r in results if "evil.jpg" in r[0]]
    assert len(found) == 1

    # (path, percent, admin_note, user_note, gpt_conf, snippet, line)
    data = found[0]
    assert data[1] == "100%" # Mismatch threat is 1.0
    assert "Deceptive content" in data[2]
    assert "Windows executable" in data[2]
