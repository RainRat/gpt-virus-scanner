import pytest
import urllib.request
from unittest.mock import MagicMock, patch
from gptscan import fetch_url_content, Config

def test_fetch_url_content_success():
    """Test that fetch_url_content succeeds when content is within max_size."""
    mock_response = MagicMock()
    mock_response.getheader.return_value = "100"
    mock_response.read.return_value = b"a" * 100
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        content = fetch_url_content("http://example.com/ok.sh")
        assert len(content) == 100
        assert content == b"a" * 100

def test_fetch_url_content_too_large_via_header():
    """Test that fetch_url_content raises ValueError when Content-Length header is too large."""
    mock_response = MagicMock()
    # 1 byte over the default 10MB
    large_size = Config.MAX_FILE_SIZE + 1
    mock_response.getheader.return_value = str(large_size)
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/large.sh")

def test_fetch_url_content_too_large_no_header():
    """
    Test that fetch_url_content raises ValueError when content exceeds max_size,
    even if the Content-Length header is missing.
    """
    mock_response = MagicMock()
    mock_response.getheader.return_value = None

    # Simulate content larger than max_size being available
    # Our mock read(n) should return up to n bytes.
    # If the bug exists, fetch_url_content will call response.read(max_size)
    # and return it without error.
    # We want it to detect that there is MORE content.

    def side_effect(size):
        if size is None or size > Config.MAX_FILE_SIZE:
            return b"a" * (Config.MAX_FILE_SIZE + 1)
        return b"a" * size

    mock_response.read.side_effect = side_effect
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/large_no_header.sh")

def test_fetch_url_content_explicit_max_size():
    """Test that fetch_url_content respects an explicitly passed max_size."""
    mock_response = MagicMock()
    mock_response.getheader.return_value = None

    custom_max = 50
    # Simulate 51 bytes available
    def side_effect(size):
        if size is None or size > custom_max:
            return b"a" * (custom_max + 1)
        return b"a" * size

    mock_response.read.side_effect = side_effect
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        # Should raise because 51 > 50
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/custom.sh", max_size=custom_max)

        # Should succeed if content is exactly custom_max
        mock_response.read.side_effect = lambda size: b"a" * min(size, custom_max)
        content = fetch_url_content("http://example.com/custom_ok.sh", max_size=custom_max)
        assert len(content) == custom_max
