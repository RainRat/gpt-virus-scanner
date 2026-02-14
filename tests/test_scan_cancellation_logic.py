import threading
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import gptscan

@pytest.fixture
def mock_scan_deps(monkeypatch):
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.1]]
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    mock_tf = MagicMock()
    mock_tf.constant = lambda x: x
    mock_tf.expand_dims = lambda x, axis: x
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    monkeypatch.setattr(gptscan.Config, "extensions_set", {".py"})

    return mock_model

def test_scan_files_cancellation_at_file_start(monkeypatch, tmp_path, mock_scan_deps):
    (tmp_path / "file1.py").write_text("content1")
    (tmp_path / "file2.py").write_text("content2")

    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [tmp_path / "file1.py", tmp_path / "file2.py"])

    cancel_event = threading.Event()

    gen = gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, cancel_event=cancel_event)

    results = []
    for event_type, data in gen:
        results.append((event_type, data))
        if event_type == 'progress' and data[2] and "file1.py" in data[2]:
            cancel_event.set()

    file_scanned = [data[0] for et, data in results if et == 'result']
    assert not any("file2.py" in f for f in file_scanned)

    progress_msgs = [str(data[2]) for et, data in results if et == 'progress' if data[2]]
    assert any("file1.py" in m for m in progress_msgs)
    assert not any("file2.py" in m for m in progress_msgs)

def test_scan_files_cancellation_during_windows(monkeypatch, tmp_path, mock_scan_deps):
    file_size = 3000
    test_file = tmp_path / "large.py"
    test_file.write_bytes(b"a" * file_size)

    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [test_file])

    cancel_event = threading.Event()

    def side_effect(*args, **kwargs):
        cancel_event.set()
        return [[0.1]]

    mock_scan_deps.predict.side_effect = side_effect

    gen = gptscan.scan_files(str(tmp_path), deep_scan=True, show_all=True, use_gpt=False, cancel_event=cancel_event)

    results = list(gen)

    assert mock_scan_deps.predict.call_count == 1
