import sys
from unittest.mock import patch
import pytest
import gptscan


def test_cli_git_diff_with_changes(monkeypatch):
    """Test main() with --git-diff flag when diff content is found."""
    mock_diff_content = "diff --git a/file.py b/file.py\n+print('hello')"
    monkeypatch.setattr(gptscan, "get_git_diff", lambda path, ref="HEAD": mock_diff_content)

    captured_extra_snippets = []

    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_extra_snippets
        captured_extra_snippets = kwargs.get("extra_snippets", [])
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-diff", "--cli"])

    gptscan.main()

    assert len(captured_extra_snippets) == 1
    filename, content = captured_extra_snippets[0]
    assert filename == "git-diff-0.patch"
    assert content == mock_diff_content.encode("utf-8")


def test_cli_git_diff_no_changes_warning(monkeypatch, capsys):
    """Test main() with --git-diff flag when no diff content is found."""
    monkeypatch.setattr(gptscan, "get_git_diff", lambda path, ref="HEAD": "")
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-diff", "--cli"])

    gptscan.main()

    captured = capsys.readouterr()
    assert "No Git diff detected in provided targets (ref: HEAD)." in captured.err


def test_cli_git_diff_custom_ref(monkeypatch):
    """Test main() with --git-diff and a custom ref (e.g., HEAD~1)."""
    captured_refs = []

    def mock_get_git_diff(path, ref="HEAD"):
        captured_refs.append(ref)
        return "diff content"

    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-diff", "HEAD~1", "--cli"])

    gptscan.main()

    assert captured_refs == ["HEAD~1"]
