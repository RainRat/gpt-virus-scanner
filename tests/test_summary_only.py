import sys
import pytest
import gptscan


def test_run_cli_summary_only_output(capsys, monkeypatch):
    """Verify that run_cli with summary_only=True outputs only the scan summary banner."""
    sample_results = [
        ('result', ('script1.py', '90%', 'Admin note 1', 'User note 1', '90%', 'exec(bad)', '10')),
        ('result', ('script2.py', '70%', 'Admin note 2', 'User note 2', '70%', 'system(bad)', '5')),
        ('summary', (2, 2048, 0.5))
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    threats = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        summary_only=True
    )

    captured = capsys.readouterr()
    out = captured.out.strip()

    assert threats == 2
    assert "Scan complete: 2 files" in out
    assert "2 suspicious files found" in out
    assert "path,own_conf" not in out
    assert "script1.py" not in out


def test_main_cli_summary_only_flag(capsys, monkeypatch):
    """Verify that passing -s or --summary-only via CLI arguments works as expected."""
    sample_results = [
        ('result', ('/path/to/bad1.py', '95%', 'Desc', 'Desc', '95%', 'snippet', '1')),
        ('summary', (1, 1024, 0.1))
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    for flag in ['-s', '--summary-only']:
        monkeypatch.setattr(sys, 'argv', ['gptscan.py', './test_folder', '--cli', '-q', flag])

        gptscan.main()
        captured = capsys.readouterr()
        out = captured.out.strip()

        assert "Scan complete: 1 file" in out
        assert "1 suspicious file found" in out
        assert "/path/to/bad1.py" not in out


def test_run_cli_summary_only_with_output_file(tmp_path, monkeypatch):
    """Verify that summary_only outputs the summary banner to a specified output file."""
    sample_results = [
        ('result', ('fileA.py', '80%', 'A', 'A', '80%', 'snip', '1')),
        ('summary', (1, 512, 0.2))
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    out_file = tmp_path / "summary.txt"

    threats = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_file=str(out_file),
        quiet=True,
        summary_only=True
    )

    assert threats == 1
    content = out_file.read_text(encoding="utf-8").strip()
    assert "Scan complete: 1 file" in content
    assert "1 suspicious file found" in content
    assert "fileA.py" not in content
