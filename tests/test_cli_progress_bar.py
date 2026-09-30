import sys
from pathlib import Path
import pytest

# Ensure root folder is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gptscan


def test_render_cli_progress_bar_basic():
    # 0%
    bar_0 = gptscan.render_cli_progress_bar(0, 100)
    assert bar_0 == "[░░░░░░░░░░]   0%"

    # 50%
    bar_50 = gptscan.render_cli_progress_bar(50, 100)
    assert bar_50 == "[█████░░░░░]  50%"

    # 100%
    bar_100 = gptscan.render_cli_progress_bar(100, 100)
    assert bar_100 == "[██████████] 100%"


def test_render_cli_progress_bar_edge_cases():
    # Zero total
    bar_zero = gptscan.render_cli_progress_bar(0, 0)
    assert bar_zero == "[░░░░░░░░░░]   0%"

    # Negative total
    bar_neg = gptscan.render_cli_progress_bar(5, -10)
    assert bar_neg == "[░░░░░░░░░░]   0%"

    # Current greater than total (clamping)
    bar_over = gptscan.render_cli_progress_bar(150, 100)
    assert bar_over == "[██████████] 100%"

    # Negative current
    bar_under = gptscan.render_cli_progress_bar(-10, 100)
    assert bar_under == "[░░░░░░░░░░]   0%"


def test_render_cli_progress_bar_custom_width():
    # Custom width = 20
    bar_custom = gptscan.render_cli_progress_bar(50, 100, width=20)
    assert bar_custom == "[██████████░░░░░░░░░░]  50%"


def test_run_cli_progress_bar_output(monkeypatch, capsys):
    def fake_scan_files(_path, _deep, _show_all, _use_gpt, _cancel_event=None, **_kwargs):
        yield ('progress', (0, 4, 'Scanning: file1.py'))
        yield ('progress', (2, 4, 'Scanning: file3.py'))
        yield ('progress', (4, 4, 'Scanning: file4.py'))

    monkeypatch.setattr(gptscan, "scan_files", fake_scan_files)

    gptscan.run_cli("/tmp", deep=False, show_all=False, use_gpt=False, rate_limit=1)

    captured = capsys.readouterr()
    assert "[░░░░░░░░░░]   0% Scanning: file1.py (0/4)" in captured.err
    assert "[█████░░░░░]  50% Scanning: file3.py (2/4)" in captured.err
    assert "[██████████] 100% Scanning: file4.py (4/4)" in captured.err
