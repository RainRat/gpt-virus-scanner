import sys
from unittest.mock import patch, MagicMock
import pytest
import argparse
from gptscan import main, Config

def test_cli_default_target(monkeypatch):
    """Test that CLI defaults to current directory if no target is provided."""
    # Mocking run_cli to avoid actual scan
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    # Simulate: python gptscan.py --cli
    test_args = ["gptscan.py", "--cli"]
    with patch.object(sys, "argv", test_args):
        try:
            main()
        except SystemExit:
            pass # argparse might exit in some cases, but here it should run

    # Verify run_cli was called with ["."] as targets
    # run_cli(targets, deep, show_all, use_gpt, rate_limit, ...)
    args, kwargs = mock_run_cli.call_args
    assert args[0] == ["."]

def test_cli_short_aliases(monkeypatch):
    """Test that short aliases for CLI flags work as expected."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    # Simulate: python gptscan.py -d -g -a -j -e "ignore_me" -p "/tmp" --cli
    test_args = [
        "gptscan.py",
        "-d",           # --deep
        "-g",           # --use-gpt
        "-a",           # --show-all
        "-j",           # --json
        "-e", "ignore_me", # --exclude
        "-p", "/tmp",   # --path
        "--cli"
    ]
    with patch.object(sys, "argv", test_args):
        try:
            main()
        except SystemExit:
            pass

    # Verify run_cli was called with correct arguments
    # targets, deep, show_all, use_gpt, rate_limit, ...
    args, kwargs = mock_run_cli.call_args

    # targets should contain "/tmp" because of -p
    assert "/tmp" in args[0]
    # deep should be True
    assert args[1] is True
    # show_all should be True
    assert args[2] is True
    # use_gpt should be True
    assert args[3] is True
    # output_format should be 'json' because of -j
    assert kwargs.get('output_format') == 'json'
    # exclude_patterns should contain "ignore_me"
    assert "ignore_me" in kwargs.get('exclude_patterns')

def test_version_alias():
    """Test that -v correctly shows version and exits."""
    test_args = ["gptscan.py", "-v"]
    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout") as mock_stdout:
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
