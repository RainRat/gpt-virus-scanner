import io
import pytest
from gptscan import iter_windows, Config

def test_iter_windows_empty_file():
    f = io.BytesIO(b"")
    windows = list(iter_windows(f, 0, deep_scan=False, maxlen=10))
    assert windows == [(0, b"")]

def test_iter_windows_shallow_small_file():
    content = b"12345"
    f = io.BytesIO(content)
    # File smaller than maxlen.
    windows = list(iter_windows(f, 5, deep_scan=False, maxlen=10))
    assert windows == [(0, b"12345")]

def test_iter_windows_shallow_exact_match():
    content = b"1234567890" # 10 bytes
    f = io.BytesIO(content)
    windows = list(iter_windows(f, 10, deep_scan=False, maxlen=10))
    assert windows == [(0, b"1234567890")]

def test_iter_windows_shallow_large_file():
    # Size 25. Maxlen 10.
    # Shallow scan logic: start and end window.
    content = b"a" * 25
    f = io.BytesIO(content)

    windows = list(iter_windows(f, 25, deep_scan=False, maxlen=10))

    assert len(windows) == 2
    assert windows[0][0] == 0
    assert len(windows[0][1]) == 10

    assert windows[1][0] == 15
    assert len(windows[1][1]) == 10

def test_iter_windows_deep_no_overlap_needed():
    # Size 20. Maxlen 10.
    content = b"a" * 20
    f = io.BytesIO(content)

    windows = list(iter_windows(f, 20, deep_scan=True, maxlen=10))

    assert len(windows) == 2
    assert windows[0][0] == 0
    assert windows[1][0] == 10

def test_iter_windows_deep_with_overlap():
    # Size 25. Maxlen 10.
    content = b"a" * 25
    f = io.BytesIO(content)

    windows = list(iter_windows(f, 25, deep_scan=True, maxlen=10))

    # 1. Offset 0. yield (0, 10 bytes)
    # 2. Offset 10. yield (10, 10 bytes)
    # 3. Offset 20. yield (20, 5 bytes)
    # 4. Overlap: last_start 15. yield (15, 10 bytes)

    assert len(windows) == 4
    assert windows[0] == (0, content[0:10])
    assert windows[1] == (10, content[10:20])
    assert windows[2] == (20, content[20:25])
    assert windows[3] == (15, content[15:25])
