import hashlib
from pathlib import Path
import pytest
import gptscan

def test_get_effective_sha256_local_file(tmp_path):
    f = tmp_path / "test.txt"
    content = b"local content"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert gptscan.get_effective_sha256(str(f)) == expected

def test_get_effective_sha256_virtual_snippet():
    snippet = "snippet content"
    expected = hashlib.sha256(snippet.encode()).hexdigest()
    # Virtual path starting with [
    assert gptscan.get_effective_sha256("[Clipboard]", snippet=snippet) == expected

def test_get_effective_sha256_missing_file_snippet():
    snippet = "snippet for missing file"
    expected = hashlib.sha256(snippet.encode()).hexdigest()
    assert gptscan.get_effective_sha256("missing.py", snippet=snippet) == expected

def test_get_effective_sha256_virtual_cache_priority(monkeypatch):
    path = "[URL] https://example.com/script.py"
    snippet = "snippet content"
    full_content = "full content from cache"

    # Populating cache
    monkeypatch.setattr(gptscan, "_virtual_source_cache", {path: full_content})

    expected_full = hashlib.sha256(full_content.encode()).hexdigest()

    # This is expected to FAIL before the fix
    assert gptscan.get_effective_sha256(path, snippet=snippet) == expected_full

def test_get_effective_sha256_none_snippet():
    assert gptscan.get_effective_sha256("[Clipboard]", snippet=None) == ""
    assert gptscan.get_effective_sha256("missing.py", snippet=None) == ""

def test_get_file_sha256_bytes():
    content = b"raw bytes"
    expected = hashlib.sha256(content).hexdigest()
    assert gptscan.get_file_sha256(content) == expected

def test_get_file_sha256_path_object(tmp_path):
    f = tmp_path / "path_obj.txt"
    content = b"path object test"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert gptscan.get_file_sha256(f) == expected
