import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import fetch_url_content

def test_fetch_url_content_truncation_detection():
    max_size = 100
    mock_response = MagicMock()
    mock_response.getheader.return_value = None
    mock_response.read.side_effect = lambda size: b"A" * size
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/truncated.sh", max_size=max_size)

def test_fetch_url_content_exact_size():
    max_size = 100
    mock_response = MagicMock()
    mock_response.getheader.return_value = None
    mock_response.read.return_value = b"A" * max_size
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        content = fetch_url_content("http://example.com/exact.sh", max_size=max_size)
        assert len(content) == max_size
