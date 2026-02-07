import pytest
from gptscan import format_scan_summary

def test_format_scan_summary_basic():
    # Basic summary without time or bytes
    summary = format_scan_summary(total_scanned=10, threats_found=2)
    assert summary == "Scan complete: 10 files scanned, 2 suspicious files found."

def test_format_scan_summary_singular():
    # Test singular "suspicious file"
    summary = format_scan_summary(total_scanned=1, threats_found=1)
    assert summary == "Scan complete: 1 files scanned, 1 suspicious file found."

def test_format_scan_summary_zero():
    # Test zero suspicious files
    summary = format_scan_summary(total_scanned=5, threats_found=0)
    assert summary == "Scan complete: 5 files scanned, 0 suspicious files found."

def test_format_scan_summary_with_time():
    # Test summary with time and files/s
    summary = format_scan_summary(total_scanned=10, threats_found=0, elapsed_time=2.0)
    assert "Time: 2.0s (5.0 files/s)." in summary

def test_format_scan_summary_with_bytes():
    # Test summary with time and throughput
    # 2048 bytes / 2.0s = 1024 bytes/s = 1.0 KiB/s
    summary = format_scan_summary(total_scanned=10, threats_found=0, total_bytes=2048, elapsed_time=2.0)
    assert "Time: 2.0s (5.0 files/s, 1.0 KiB/s)." in summary

def test_format_scan_summary_zero_time():
    # Test that zero time does not cause division by zero and omits metrics
    summary = format_scan_summary(total_scanned=10, threats_found=0, elapsed_time=0.0)
    assert "Time:" not in summary
    assert summary == "Scan complete: 10 files scanned, 0 suspicious files found."

def test_format_scan_summary_none_bytes():
    # Test that None bytes omits throughput
    summary = format_scan_summary(total_scanned=10, threats_found=0, total_bytes=None, elapsed_time=2.0)
    assert "5.0 files/s" in summary
    assert "KiB/s" not in summary

def test_format_scan_summary_large_throughput():
    # 10 MiB / 1s = 10 MiB/s
    size = 10 * 1024 * 1024
    summary = format_scan_summary(total_scanned=1, threats_found=0, total_bytes=size, elapsed_time=1.0)
    assert "10.0 MiB/s" in summary
