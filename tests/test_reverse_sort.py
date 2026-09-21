import io
import os
import sys
from unittest.mock import MagicMock
import pytest

import gptscan


def test_reverse_sort_threat(monkeypatch):
    """Test -r / --reverse with --sort-by threat in run_cli."""
    out_stream = io.StringIO()
    records = [
        {"path": "low.py", "own_conf": "20%", "line": "10"},
        {"path": "high.py", "own_conf": "90%", "line": "5"},
        {"path": "med.py", "own_conf": "50%", "line": "1"},
    ]

    # Mock scan_files event generator
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 3, "Scanning"))
        yield ('result', ("low.py", "20%", "desc", "desc", "", "snippet", "10"))
        yield ('result', ("high.py", "90%", "desc", "desc", "", "snippet", "5"))
        yield ('result', ("med.py", "50%", "desc", "desc", "", "snippet", "1"))
        yield ('summary', (3, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)
    monkeypatch.setattr(gptscan, "_get_initial_dir", lambda: None)
    monkeypatch.setattr(sys, "stdout", out_stream)

    # Standard threat sort (highest threat first: high.py, med.py, low.py)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="threat",
        reverse_sort=False,
        quiet=True
    )
    lines = [line.strip() for line in out_stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    assert '"path": "high.py"' in lines[0]
    assert '"path": "med.py"' in lines[1]
    assert '"path": "low.py"' in lines[2]

    # Reverse threat sort (lowest threat first: low.py, med.py, high.py)
    out_stream_rev = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream_rev)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="threat",
        reverse_sort=True,
        quiet=True
    )
    lines_rev = [line.strip() for line in out_stream_rev.getvalue().splitlines() if line.strip()]
    assert len(lines_rev) == 3
    assert '"path": "low.py"' in lines_rev[0]
    assert '"path": "med.py"' in lines_rev[1]
    assert '"path": "high.py"' in lines_rev[2]


def test_reverse_sort_path(monkeypatch):
    """Test -r / --reverse with --sort-by path in run_cli."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 3, "Scanning"))
        yield ('result', ("b.py", "80%", "desc", "desc", "", "snippet", "1"))
        yield ('result', ("a.py", "80%", "desc", "desc", "", "snippet", "1"))
        yield ('result', ("c.py", "80%", "desc", "desc", "", "snippet", "1"))
        yield ('summary', (3, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Standard path sort (a.py, b.py, c.py)
    out_stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="path",
        reverse_sort=False,
        quiet=True
    )
    lines = [line.strip() for line in out_stream.getvalue().splitlines() if line.strip()]
    assert '"path": "a.py"' in lines[0]
    assert '"path": "b.py"' in lines[1]
    assert '"path": "c.py"' in lines[2]

    # Reverse path sort (c.py, b.py, a.py)
    out_stream_rev = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream_rev)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="path",
        reverse_sort=True,
        quiet=True
    )
    lines_rev = [line.strip() for line in out_stream_rev.getvalue().splitlines() if line.strip()]
    assert '"path": "c.py"' in lines_rev[0]
    assert '"path": "b.py"' in lines_rev[1]
    assert '"path": "a.py"' in lines_rev[2]


def test_reverse_sort_line(monkeypatch):
    """Test -r / --reverse with --sort-by line in run_cli."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 3, "Scanning"))
        yield ('result', ("file.py", "80%", "desc", "desc", "", "snippet", "100"))
        yield ('result', ("file.py", "80%", "desc", "desc", "", "snippet", "5"))
        yield ('result', ("file.py", "80%", "desc", "desc", "", "snippet", "42"))
        yield ('summary', (3, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Standard line sort (5, 42, 100)
    out_stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="line",
        reverse_sort=False,
        quiet=True
    )
    lines = [line.strip() for line in out_stream.getvalue().splitlines() if line.strip()]
    assert '"line": "5"' in lines[0]
    assert '"line": "42"' in lines[1]
    assert '"line": "100"' in lines[2]

    # Reverse line sort (100, 42, 5)
    out_stream_rev = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream_rev)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by="line",
        reverse_sort=True,
        quiet=True
    )
    lines_rev = [line.strip() for line in out_stream_rev.getvalue().splitlines() if line.strip()]
    assert '"line": "100"' in lines_rev[0]
    assert '"line": "42"' in lines_rev[1]
    assert '"line": "5"' in lines_rev[2]


def test_reverse_sort_default_threat(monkeypatch):
    """Test -r / --reverse without explicit --sort-by reverses default threat-based sorting."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 2, "Scanning"))
        yield ('result', ("low.py", "30%", "desc", "desc", "", "snippet", "1"))
        yield ('result', ("high.py", "95%", "desc", "desc", "", "snippet", "1"))
        yield ('summary', (2, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Reverse default sorting (lowest threat first)
    out_stream_rev = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_stream_rev)
    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        sort_by=None,
        reverse_sort=True,
        quiet=True
    )
    lines_rev = [line.strip() for line in out_stream_rev.getvalue().splitlines() if line.strip()]
    assert '"path": "low.py"' in lines_rev[0]
    assert '"path": "high.py"' in lines_rev[1]
