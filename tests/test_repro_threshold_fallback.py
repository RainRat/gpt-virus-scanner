import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
import gptscan

@pytest.fixture
def mock_scan_dependencies(monkeypatch):
    """Mocks TensorFlow, Model, and file system for scan_files tests."""
    mock_predict = MagicMock(return_value=[[0.1]]) # 10% threat
    mock_model = SimpleNamespace(predict=mock_predict)
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    mock_tf = SimpleNamespace(
        constant=lambda x: x,
        expand_dims=lambda x, axis: x,
    )
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)
    return mock_predict

def test_scan_files_fallback_when_below_threshold(monkeypatch, tmp_path, mock_scan_dependencies):
    """
    Verify that if no segments exceed the threshold, it still falls back to the best hit
    even if show_all is False (as per documentation).
    """
    test_file = tmp_path / "below_threshold.py"
    test_file.write_bytes(b"print('hello')")

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [test_file])
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 50)

    # Run scan with show_all=False
    results = list(gptscan.scan_files(
        scan_targets=str(tmp_path),
        deep_scan=False,
        show_all=False,
        use_gpt=False
    ))

    result_events = [r for r in results if r[0] == 'result']

    # This is EXPECTED TO FAIL currently (len will be 0)
    assert len(result_events) == 1, "Should yield the best hit even if below threshold when show_all is False"
    assert result_events[0][1][1] == "10%"
