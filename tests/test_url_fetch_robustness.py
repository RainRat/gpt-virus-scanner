import pytest
from unittest.mock import MagicMock, patch
from gptscan import fetch_url_content, Config

def test_fetch_url_content_detects_truncation():
    max_size = 100
    mock_response = MagicMock()
    mock_response.getheader.return_value = None

    def mock_read(size):
        if size == max_size + 1:
            return b"a" * (max_size + 1)
        return b"a" * min(size, max_size + 1)

    mock_response.read.side_effect = mock_read
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        # This is expected to FAIL (not raise ValueError) with current implementation
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/toolarge.py", max_size=max_size)

def test_fetch_url_content_with_none_max_size():
    original_max = Config.MAX_FILE_SIZE
    Config.MAX_FILE_SIZE = 50
    try:
        mock_response = MagicMock()
        mock_response.getheader.return_value = "100" # Larger than 50
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="Content too large"):
                fetch_url_content("http://example.com/default_limit.py", max_size=None)
    finally:
        Config.MAX_FILE_SIZE = original_max

def test_fetch_url_content_success_within_limit():
    max_size = 100
    content = b"a" * max_size

    mock_response = MagicMock()
    mock_response.getheader.return_value = None
    mock_response.read.return_value = content
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = fetch_url_content("http://example.com/ok.py", max_size=max_size)
        assert len(result) == max_size
        assert result == content
