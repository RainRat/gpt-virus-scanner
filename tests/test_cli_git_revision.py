"""Tests for --git-revision CLI flag in gptscan."""

from unittest.mock import patch
import gptscan


def test_cli_git_revision_default():
    """Verify that --git-revision without an explicit value sets ref to 'HEAD'."""
    captured_ref = []

    def mock_get_git_changed_files(path=".", ref="HEAD"):
        captured_ref.append(ref)
        return ["main.py"]

    test_args = ["gptscan.py", "--git-revision", "--cli"]
    with patch.object(gptscan.sys, "argv", test_args):
        with patch.object(gptscan, "get_git_changed_files", side_effect=mock_get_git_changed_files):
            with patch("gptscan.run_cli", return_value=0) as mock_run_cli:
                try:
                    gptscan.main()
                except SystemExit:
                    pass
                assert "HEAD" in captured_ref
                assert mock_run_cli.called
                positional_targets = mock_run_cli.call_args[0][0]
                assert positional_targets == ["main.py"]


def test_cli_git_revision_explicit_ref(monkeypatch):
    """Verify that --git-revision main correctly passes 'main' as the git revision ref."""
    captured_ref = []

    def mock_get_git_changed_files(path=".", ref="HEAD"):
        captured_ref.append(ref)
        return ["file1.py", "file2.py"]

    monkeypatch.setattr(gptscan, "get_git_changed_files", mock_get_git_changed_files)

    test_args = ["gptscan.py", "--git-revision", "main", "--cli"]
    with patch.object(gptscan.sys, "argv", test_args):
        with patch("gptscan.run_cli", return_value=0) as mock_run_cli:
            try:
                gptscan.main()
            except SystemExit:
                pass
            assert "main" in captured_ref
            assert mock_run_cli.called
            positional_targets = mock_run_cli.call_args[0][0]
            assert positional_targets == ["file1.py", "file2.py"]
