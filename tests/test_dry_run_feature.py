import pytest
from unittest.mock import MagicMock
import gptscan

def test_dry_run_skips_model_and_yields_results(monkeypatch, tmp_path):
    """
    Verify that when dry_run=True:
    1. The model is NOT loaded or used.
    2. Files are found and yielded with 'Dry Run' status.
    3. Progress events are yielded.
    """
    # Setup files
    f1 = tmp_path / "test1.py"
    f1.write_text("print('hello')")
    f2 = tmp_path / "test2.js"
    f2.write_text("console.log('hello')")

    # Mock Config extensions to include our files
    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py", ".js"})

    # Mock collect_files to return these files explicitly
    monkeypatch.setattr(gptscan, "collect_files", lambda x: [f1, f2])

    # Mock get_model to ensure it's NOT called
    mock_get_model = MagicMock()
    monkeypatch.setattr(gptscan, "get_model", mock_get_model)

    # Run scan_files with dry_run=True
    events = list(gptscan.scan_files(
        scan_targets=str(tmp_path),
        deep_scan=False,
        show_all=False,
        use_gpt=False,
        dry_run=True
    ))

    # Assertions

    # 1. Model should not be accessed
    mock_get_model.assert_not_called()

    # 2. Check results
    results = [e for e in events if e[0] == 'result']
    assert len(results) == 2, "Should yield a result for each file"

    for _, data in results:
        path, status, admin, user, gpt, snippet = data
        assert status == "Dry Run"
        assert snippet == "(File would be scanned)"
        # Confirm no analysis data
        assert admin == ""
        assert user == ""
        assert gpt == ""

    # 3. Check progress
    progress_events = [e for e in events if e[0] == 'progress']
    # Initial "Scanning..." + 2 updates
    assert len(progress_events) >= 3
    assert progress_events[0][1][2] == "Scanning..."
