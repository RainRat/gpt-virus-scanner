import pytest
from gptscan import fetch_url_content

def test_fetch_url_content_unauthorized_scheme():
    """Verify that fetch_url_content rejects non-HTTP(S) schemes."""
    with pytest.raises(ValueError, match="Unauthorized URL scheme"):
        fetch_url_content("file:///etc/passwd")

    with pytest.raises(ValueError, match="Unauthorized URL scheme"):
        fetch_url_content("ftp://example.com/file")

    with pytest.raises(ValueError, match="Unauthorized URL scheme"):
        fetch_url_content("gopher://example.com")

def test_fetch_url_content_authorized_scheme(mocker):
    """Verify that fetch_url_content allows HTTP(S) schemes."""
    # Mock urllib.request.urlopen to avoid actual network requests
    # Use MagicMock for context manager support
    mock_response = mocker.MagicMock()
    mock_response.getheader.return_value = "100"
    mock_response.read.return_value = b"test content"
    mock_response.__enter__.return_value = mock_response

    mocker.patch("urllib.request.urlopen", return_value=mock_response)

    # These should not raise ValueError for the scheme
    fetch_url_content("http://example.com")
    fetch_url_content("https://example.com")
