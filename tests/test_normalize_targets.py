import pytest
from pathlib import Path
from gptscan import _normalize_targets

def test_normalize_targets_path():
    p = Path("/tmp/test")
    assert _normalize_targets(p) == ["/tmp/test"]

def test_normalize_targets_string():
    s = "/tmp/a /tmp/b"
    assert _normalize_targets(s) == ["/tmp/a", "/tmp/b"]

def test_normalize_targets_string_quoted():
    s = "'/tmp/path with space' /tmp/b"
    assert _normalize_targets(s) == ["/tmp/path with space", "/tmp/b"]

def test_normalize_targets_string_malformed_quotes():
    s = "'malformed quote"
    # Fallback to returning the string as a single-element list
    assert _normalize_targets(s) == [s]

def test_normalize_targets_list_strings():
    l = ["/tmp/a", "/tmp/b", "/tmp/a"]
    assert _normalize_targets(l) == ["/tmp/a", "/tmp/b"]

def test_normalize_targets_list_paths():
    l = [Path("/tmp/a"), Path("/tmp/b"), Path("/tmp/a")]
    # Current implementation fails this: it returns [Path("/tmp/a"), Path("/tmp/b"), Path("/tmp/a")]
    # Expected: ["/tmp/a", "/tmp/b"]
    result = _normalize_targets(l)
    assert result == ["/tmp/a", "/tmp/b"]
    for item in result:
        assert isinstance(item, str)

def test_normalize_targets_mixed_list():
    l = [Path("/tmp/a"), "/tmp/a", "/tmp/b"]
    # Current implementation fails this: [Path("/tmp/a"), "/tmp/a", "/tmp/b"]
    # Expected: ["/tmp/a", "/tmp/b"]
    result = _normalize_targets(l)
    assert result == ["/tmp/a", "/tmp/b"]
    for item in result:
        assert isinstance(item, str)

def test_normalize_targets_empty_list():
    assert _normalize_targets([]) == []

def test_normalize_targets_preserves_order():
    l = ["b", "a", "b", "c"]
    assert _normalize_targets(l) == ["b", "a", "c"]
