import pytest
from unittest.mock import MagicMock, patch
from gptscan import fetch_url_content, Config

def test_fetch_url_content_truncation_detection():
    """Test that fetch_url_content detects truncation when Content-Length is missing."""
    max_size = 100
    # Simulate content larger than max_size
    mock_content = b"a" * (max_size + 1)

    mock_response = MagicMock()
    mock_response.getheader.return_value = None # No content-length
    # Mock read to return max_size + 1 bytes when asked
    mock_response.read.side_effect = lambda size: mock_content[:size]
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/oversized", max_size=max_size)

def test_fetch_url_content_within_limit_no_header():
    """Test that fetch_url_content works correctly within limit even if Content-Length is missing."""
    max_size = 100
    mock_content = b"a" * 50

    mock_response = MagicMock()
    mock_response.getheader.return_value = None # No content-length
    mock_response.read.side_effect = lambda size: mock_content[:size]
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        content = fetch_url_content("http://example.com/valid", max_size=max_size)
        assert content == mock_content
        assert len(content) == 50
