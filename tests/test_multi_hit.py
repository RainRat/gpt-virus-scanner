import io
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
import gptscan

@pytest.fixture
def mock_multi_hit_model(monkeypatch):
    """Mocks the model to return specific threat levels for different windows."""

    def mock_predict(tf_data, batch_size=1, steps=1):
        # Extract the first byte of the data to decide the return value
        # tf_data is a tensor, we assume it's passed as a list or similar in mock
        data = tf_data.data if hasattr(tf_data, 'data') else tf_data
        first_byte = data[0]
        if first_byte == ord('H'): # 'H' for High threat
            return [[0.9]]
        return [[0.1]]

    mock_model = MagicMock()
    mock_model.predict.side_effect = mock_predict
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    # Mock TensorFlow module
    mock_tf = SimpleNamespace(
        constant=lambda x: SimpleNamespace(data=x),
        expand_dims=lambda x, axis: x,
    )
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    return mock_model

def test_scan_files_reports_multiple_hits_in_one_file(monkeypatch, tmp_path, mock_multi_hit_model):
    """Verify that multiple suspicious segments in one file are all reported."""
    # Create a file with two suspicious segments
    # Segment 1 starts at 0 with 'H', Segment 2 starts at 1024 with 'H'
    content = b'H' + b'a' * 1023 + b'H' + b'b' * 1023
    test_file = tmp_path / "multi_hit.py"
    test_file.write_bytes(content)

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [test_file])

    # Run scan with deep_scan=True to find both windows
    results = list(gptscan.scan_files(
        scan_targets=str(tmp_path),
        deep_scan=True,
        show_all=False,
        use_gpt=False
    ))

    # Expect: Progress (start), Progress (file), Result (Hit 1), Result (Hit 2), Summary
    result_events = [r for r in results if r[0] == 'result']
    assert len(result_events) == 2

    # Verify Hit 1 (offset 0 -> line 1)
    assert result_events[0][1][1] == "90%"
    assert result_events[0][1][6] == 1

    # Verify Hit 2 (offset 1024 -> line 1 because no newline)
    assert result_events[1][1][6] == 1

    # Test with newline to verify line counting
    # 1023 bytes at 0...1022. Then '\n' at 1023. Then 'H' at 1024.
    content_with_newline = b'H' + b'a' * 1022 + b'\n' + b'H' + b'b' * 1023
    test_file.write_bytes(content_with_newline)

    results = list(gptscan.scan_files(
        scan_targets=str(tmp_path),
        deep_scan=True,
        show_all=False,
        use_gpt=False
    ))

    result_events = [r for r in results if r[0] == 'result']
    assert len(result_events) == 2
    assert result_events[0][1][6] == 1
    assert result_events[1][1][6] == 2 # Second hit is after the newline

def test_scan_files_extra_snippets_multiple_hits(monkeypatch, mock_multi_hit_model):
    """Verify that multiple suspicious segments in an extra snippet are all reported."""
    # Hit 1 at 0. '\n' at 1023. Hit 2 at 1024.
    content = b'H' + b'a' * 1022 + b'\n' + b'H' + b'b' * 1023
    extra_snippets = [("[Clipboard]", content)]

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})

    # Run scan
    results = list(gptscan.scan_files(
        scan_targets=[],
        deep_scan=True,
        show_all=False,
        use_gpt=False,
        extra_snippets=extra_snippets
    ))

    result_events = [r for r in results if r[0] == 'result']
    assert len(result_events) == 2
    assert result_events[0][1][0] == "[Clipboard]"
    assert result_events[0][1][6] == 1
    assert result_events[1][1][6] == 2

def test_scan_files_fallback_if_no_multi_hits(monkeypatch, tmp_path, mock_multi_hit_model):
    """Verify that if no segments exceed the threshold, it still falls back to the best hit (if show_all)."""
    # File with only low threat segments (not starting with 'H')
    content = b'low threat'
    test_file = tmp_path / "safe.py"
    test_file.write_bytes(content)

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [test_file])

    # Run scan with show_all=True
    results = list(gptscan.scan_files(
        scan_targets=str(tmp_path),
        deep_scan=True,
        show_all=True,
        use_gpt=False
    ))

    result_events = [r for r in results if r[0] == 'result']
    assert len(result_events) == 1
    assert result_events[0][1][1] == "10%"
