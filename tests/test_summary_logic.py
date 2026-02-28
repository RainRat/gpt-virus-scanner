import gptscan
from unittest.mock import MagicMock
import sys
import io

def test_summary_display(monkeypatch):
    # Mock data
    total_scanned = 10
    threats_found = 2
    total_bytes = 1024 * 1024 * 5.5 # 5.5 MiB
    elapsed_time = 2.0
    # Expected throughput: 5.5 MiB / 2.0s = 2.75 MiB/s -> 2.8 MiB/s formatted

    # GUI Test
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, "status_label", mock_status_label)
    
    gptscan.finish_scan_state(total_scanned, threats_found, total_bytes, elapsed_time)
    
    called_msg = mock_status_label.config.call_args[1]['text']
    assert "2.8 MiB/s" in called_msg
    assert "5.0 files/s" in called_msg

    # CLI Test
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (10, 10, None))
        yield ('summary', (10, int(total_bytes), elapsed_time))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    stderr_mock = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_mock)

    gptscan.run_cli(["."], False, True, False, 60)

    output = stderr_mock.getvalue()
    assert "2.8 MiB/s" in output
    assert "5.0 files/s" in output
