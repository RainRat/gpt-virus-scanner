import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import gptscan

# Verify deep scan logic which was previously uncovered.
# This ensures that files are chunked correctly and the overlapping end window is processed.

@pytest.fixture
def mock_scan_dependencies(monkeypatch):
    """Mocks TensorFlow, Model, and file system for scan_files tests."""

    # Mock model prediction to return low confidence (avoiding GPT calls)
    mock_predict = MagicMock(return_value=[[0.1]])
    mock_model = SimpleNamespace(predict=mock_predict)

    # Ensure get_model returns our mock
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    # Mock TensorFlow tensor behavior
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, key):
             return self

    # Mock TensorFlow module
    mock_tf = SimpleNamespace(
        constant=lambda x: x,
        expand_dims=lambda x, axis: x,
        keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda p, compile=False: mock_model))
    )
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    return mock_predict

@pytest.mark.parametrize("file_size, expected_calls", [
    (500, 1),           # < MAXLEN (1024) -> 1 call (padded)
    (1024, 1),          # = MAXLEN -> 1 call
    (2048, 2),          # = 2 * MAXLEN -> 2 calls
    (2500, 4),          # = 2.44 * MAXLEN -> 0, 1024, 2048(partial), 1476(overlap) -> 4 calls
])
def test_scan_files_deep_scan_coverage(monkeypatch, tmp_path, mock_scan_dependencies, file_size, expected_calls):
    # Setup: Create a test file of specific size
    file_content = b"a" * file_size
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(file_content)

    # Mock Config to include our file extension
    # Using a set since the code checks: if extension in Config.extensions_set
    monkeypatch.setattr(gptscan.Config, "extensions_set", {".bin"})

    # Mock list_files to return only our test file
    monkeypatch.setattr(gptscan, "list_files", lambda p: [test_file])

    # Execute scan with deep_scan=True
    # Convert generator to list to ensure execution
    list(gptscan.scan_files(
        scan_path=str(tmp_path),
        deep_scan=True,
        show_all=True,
        use_gpt=False
    ))

    # Verify the model was called the expected number of times
    assert mock_scan_dependencies.call_count == expected_calls


def test_scan_files_handles_permission_error(monkeypatch, tmp_path, mock_scan_dependencies):
    # Setup
    test_file = tmp_path / "protected.bin"
    test_file.write_bytes(b"secret")

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".bin"})
    monkeypatch.setattr(gptscan, "list_files", lambda p: [test_file])

    # Mock open to raise PermissionError
    def mock_open(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr("builtins.open", mock_open)

    results = list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False))

    # Verify we get an Error result
    # Result format: ('result', (path, 'Error', '', '', '', error_msg))
    # results[0] is progress (0, 1, None)
    # results[1] is result
    # results[2] is progress (1, 1, None)
    assert len(results) == 3
    res_type, res_data = results[1]
    assert res_type == 'result'
    assert res_data[1] == 'Error'
    assert "Access denied" in res_data[5]
