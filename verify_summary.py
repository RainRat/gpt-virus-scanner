
import gptscan
from unittest.mock import MagicMock
import sys

def test_summary_display():
    # Mock data
    total_scanned = 10
    threats_found = 2
    total_bytes = 1024 * 1024 * 5.5 # 5.5 MB
    elapsed_time = 2.0

    print("Testing finish_scan_state (GUI)...")
    gptscan.status_label = MagicMock()
    gptscan.finish_scan_state(total_scanned, threats_found, total_bytes, elapsed_time)
    called_msg = gptscan.status_label.config.call_args[1]['text']
    print(f"GUI Status: {called_msg}")
    assert "5.5 MiB/s" in called_msg
    assert "5.0 files/s" in called_msg

    print("\nTesting run_cli summary (CLI)...")
    # We can't easily run run_cli but we can test the summary printing logic if we extract it
    # Or just mock scan_files to yield the summary

    def mock_scan_files(*args, **kwargs):
        yield ('progress', (10, 10, None))
        yield ('summary', (10, total_bytes, elapsed_time))

    gptscan.scan_files = mock_scan_files

    import io
    stderr_mock = io.StringIO()
    sys.stderr = stderr_mock

    try:
        gptscan.run_cli(["."], False, True, False, 60)
    finally:
        sys.stderr = sys.__stderr__

    output = stderr_mock.getvalue()
    print(f"CLI Stderr:\n{output}")
    assert "5.5 MiB/s" in output
    assert "5.0 files/s" in output

if __name__ == "__main__":
    test_summary_display()
