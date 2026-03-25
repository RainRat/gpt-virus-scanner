import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import gptscan

def test_parse_size_string_comprehensive():
    # All confirmed units
    assert gptscan.parse_size_string("1B") == 1
    assert gptscan.parse_size_string("1K") == 1024
    assert gptscan.parse_size_string("1KB") == 1024
    assert gptscan.parse_size_string("1KIB") == 1024
    assert gptscan.parse_size_string("1M") == 1024**2
    assert gptscan.parse_size_string("1MB") == 1024**2
    assert gptscan.parse_size_string("1MIB") == 1024**2
    assert gptscan.parse_size_string("1G") == 1024**3
    assert gptscan.parse_size_string("1GB") == 1024**3
    assert gptscan.parse_size_string("1GIB") == 1024**3
    assert gptscan.parse_size_string("1T") == 1024**4
    assert gptscan.parse_size_string("1TB") == 1024**4
    assert gptscan.parse_size_string("1TIB") == 1024**4

    # Case insensitivity and whitespace
    assert gptscan.parse_size_string("10mb") == 10 * 1024**2
    assert gptscan.parse_size_string(" 10 MB ") == 10 * 1024**2

    # Decimals
    assert gptscan.parse_size_string("1.5MB") == int(1.5 * 1024**2)

    # Bare numbers
    assert gptscan.parse_size_string("100") == 100

    # Error cases
    with pytest.raises(ValueError, match="Size string is empty"):
        gptscan.parse_size_string("")

    with pytest.raises(ValueError, match="Invalid size format"):
        gptscan.parse_size_string("invalid")

    with pytest.raises(ValueError, match="Unknown unit: XB"):
        gptscan.parse_size_string("10XB")

def test_fetch_url_content_size_limit():
    with patch("urllib.request.urlopen") as mock_url:
        mock_response = MagicMock()
        mock_response.getheader.return_value = "20000000" # 20MB
        mock_url.return_value.__enter__.return_value = mock_response

        gptscan.Config.MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB

        with pytest.raises(ValueError, match="Content too large"):
            gptscan.fetch_url_content("http://example.com/large.py")

def test_scan_files_skips_large_local_file(tmp_path, monkeypatch):
    large_file = tmp_path / "large.py"
    large_file.write_text("print('hello')" * 1000)

    # Set limit very low
    monkeypatch.setattr(gptscan.Config, 'MAX_FILE_SIZE', 100)

    # Mock model to return something
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.9]]
    monkeypatch.setattr(gptscan, 'get_model', lambda: mock_model)
    monkeypatch.setattr(gptscan, '_tf_module', MagicMock())

    # Scan the directory, so the file is not "explicitly" requested
    gen = gptscan.scan_files([str(tmp_path)], deep_scan=False, show_all=True, use_gpt=False)

    results = []
    for event_type, data in gen:
        if event_type == 'result':
            results.append(data)

    assert len(results) == 1
    assert results[0][1] == 'Large File'
    assert "exceeds maximum size" in results[0][5]

def test_unpack_content_respects_size_limit(monkeypatch):
    # Mock Config.MAX_FILE_SIZE
    monkeypatch.setattr(gptscan.Config, 'MAX_FILE_SIZE', 100)

    # Create a small "archive" content that would yield a member
    # Instead of real zip, I'll mock zipfile.ZipFile
    with patch("zipfile.ZipFile") as mock_zip:
        mock_info = MagicMock()
        mock_info.is_dir.return_value = False
        mock_info.file_size = 200 # Larger than limit
        mock_info.filename = "large_member.py"

        mock_zip.return_value.__enter__.return_value.infolist.return_value = [mock_info]

        content = b"PK\x03\x04 fake zip content"
        gen = gptscan.unpack_content("test.zip", content)

        # Should yield nothing because member is too large
        results = list(gen)
        assert len(results) == 0
