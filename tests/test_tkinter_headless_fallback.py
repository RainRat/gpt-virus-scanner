import sys
from unittest.mock import patch, MagicMock
import pytest
import gptscan
from gptscan import main, Config, Dummy

def test_tkinter_not_available_fallback_to_cli(monkeypatch, capsys):
    """Test that if TK_AVAILABLE is False, main() gracefully falls back to CLI mode and alerts the user."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)
    monkeypatch.setattr("gptscan.TK_AVAILABLE", False)

    # Simulate: running without --cli or any CLI-specific flags, so it tries GUI mode
    test_args = ["gptscan.py"]
    with patch.object(sys, "argv", test_args):
        try:
            main()
        except SystemExit:
            pass

    # Verify run_cli was called (graceful fallback)
    assert mock_run_cli.called

    # Verify the warning was printed to stderr
    captured = capsys.readouterr()
    assert "Warning: tkinter is not installed. Falling back to terminal (CLI) mode." in captured.err

def test_dummy_behavior():
    """Test that the Dummy mock behaves correctly, returning subclassable entities and handling calls."""
    d = Dummy()
    # Getting attributes should return the Dummy class so it is subclassable
    attr = d.some_attribute
    assert attr.__name__ == "Dummy"

    # Calling it should return an instance of Dummy
    instance = d()
    assert isinstance(instance, Dummy)

    # Subclassing a nested attribute should not raise TypeError
    class TestClass(d.Frame):
        pass
    assert TestClass is not None
