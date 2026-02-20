import pytest
from gptscan import run_cli
from pathlib import Path
import os
import json

def test_run_cli_output_file(tmp_path):
    output_file = tmp_path / "results.json"
    targets = [str(Path(__file__).parent / "test_data")]
    # Mocking scan_files to return something
    # But run_cli calls scan_files, which we might want to mock if we don't want real scanning
    # For now, let's use a simple target and dry_run

    run_cli(
        targets=["gptscan.py"],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        dry_run=True,
        output_file=str(output_file)
    )

    assert output_file.exists()
    with open(output_file, "r") as f:
        content = f.read()
        assert content.strip() != ""
        # Verify it's JSON
        json.loads(content.splitlines()[0])

def test_inference_logic(monkeypatch, tmp_path):
    import gptscan
    import sys

    output_file = tmp_path / "report.html"

    # We want to test the logic in main() that sets output_format
    # Since main() uses argparse, we can mock sys.argv
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--cli", "-o", str(output_file), "gptscan.py", "--dry-run"])

    # We need to catch SystemExit because main() might call sys.exit()
    # or just call main() and see if it calls run_cli with correct args.

    recorded_args = {}
    def mock_run_cli(*args, **kwargs):
        recorded_args['output_format'] = kwargs.get('output_format')
        recorded_args['output_file'] = kwargs.get('output_file')
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    # Also need to mock create_gui and mainloop to avoid opening windows if --cli is missed

    gptscan.main()

    assert recorded_args['output_format'] == 'html'
    assert recorded_args['output_file'] == str(output_file)

def test_inference_logic_csv(monkeypatch, tmp_path):
    import gptscan
    import sys

    output_file = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--cli", "-o", str(output_file), "gptscan.py", "--dry-run"])

    recorded_args = {}
    def mock_run_cli(*args, **kwargs):
        recorded_args['output_format'] = kwargs.get('output_format')
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    gptscan.main()

    assert recorded_args['output_format'] == 'csv'
