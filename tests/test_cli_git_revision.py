import sys
import pytest
import gptscan


def test_cli_git_revision_default_ref(monkeypatch):
    """Test main() with --git-revision flag defaulting to HEAD."""
    mock_changed_files = ["src/app.py", "scripts/util.py"]
    captured_refs = []

    def mock_get_git_changed_files(path=".", ref="HEAD"):
        captured_refs.append((path, ref))
        return mock_changed_files

    monkeypatch.setattr(gptscan, "get_git_changed_files", mock_get_git_changed_files)

    captured_targets = []

    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_targets
        captured_targets = targets
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-revision", "--cli"])

    gptscan.main()

    assert captured_refs == [(".", "HEAD")]
    assert captured_targets == mock_changed_files


def test_cli_git_revision_custom_ref(monkeypatch):
    """Test main() with --git-revision and a custom ref (e.g. main)."""
    captured_refs = []

    def mock_get_git_changed_files(path=".", ref="HEAD"):
        captured_refs.append((path, ref))
        return ["main.py"]

    monkeypatch.setattr(gptscan, "get_git_changed_files", mock_get_git_changed_files)

    captured_targets = []

    def mock_run_cli(targets, *args, **kwargs):
        nonlocal captured_targets
        captured_targets = targets
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-revision", "main", "--cli"])

    gptscan.main()

    assert captured_refs == [(".", "main")]
    assert captured_targets == ["main.py"]


def test_cli_git_revision_no_changes_warning(monkeypatch, capsys):
    """Test main() with --git-revision when no changed files are detected."""
    monkeypatch.setattr(gptscan, "get_git_changed_files", lambda path=".", ref="HEAD": [])
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--git-revision", "v1.0.0", "--cli"])

    gptscan.main()

    captured = capsys.readouterr()
    assert "No git changes detected in provided targets (ref: v1.0.0)." in captured.err


def test_cli_git_revision_custom_target(monkeypatch):
    """Test main() with custom target path and --git-revision flag."""
    captured_refs = []

    def mock_get_git_changed_files(path=".", ref="HEAD"):
        captured_refs.append((path, ref))
        return ["my_folder/file.py"]

    monkeypatch.setattr(gptscan, "get_git_changed_files", mock_get_git_changed_files)
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 0)

    monkeypatch.setattr("sys.argv", ["gptscan.py", "my_folder/", "--git-revision", "HEAD~2", "--cli"])

    gptscan.main()

    assert captured_refs == [("my_folder/", "HEAD~2")]
