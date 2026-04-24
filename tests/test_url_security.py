import pytest
from unittest.mock import MagicMock, patch
from gptscan import fetch_url_content

def test_fetch_url_content_allowed_schemes():
    mock_response = MagicMock()
    mock_response.getheader.return_value = "10"
    mock_response.read.return_value = b"test content"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        # Test http
        content = fetch_url_content("http://example.com/script.py")
        assert content == b"test content"
        mock_urlopen.assert_called_with("http://example.com/script.py", timeout=10)

        # Test https
        content = fetch_url_content("https://example.com/script.py")
        assert content == b"test content"
        mock_urlopen.assert_called_with("https://example.com/script.py", timeout=10)

def test_fetch_url_content_unauthorized_schemes(tmp_path):
    # Test file://
    secrets = tmp_path / "secrets.txt"
    secrets.write_text("secret")
    file_url = f"file://{secrets.absolute()}"

    with pytest.raises(ValueError, match="Unsupported URL scheme: file"):
        fetch_url_content(file_url)

    # Test ftp://
    with pytest.raises(ValueError, match="Unsupported URL scheme: ftp"):
        fetch_url_content("ftp://example.com/file")

    # Test data://
    with pytest.raises(ValueError, match="Unsupported URL scheme: data"):
        fetch_url_content("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")

def test_fetch_url_content_case_insensitive_scheme():
    mock_response = MagicMock()
    mock_response.getheader.return_value = "10"
    mock_response.read.return_value = b"test"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        content = fetch_url_content("HTTP://example.com/script.py")
        assert content == b"test"

        content = fetch_url_content("Https://example.com/script.py")
        assert content == b"test"
