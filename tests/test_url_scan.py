import io
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import scan_files, Config, fetch_url_content

def test_fetch_url_content_success():
    """Test successful URL content fetching."""
    mock_response = MagicMock()
    mock_response.getheader.return_value = "100"
    mock_response.read.return_value = b"some content"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        content = fetch_url_content("http://example.com/script.sh")
        assert content == b"some content"

def test_fetch_url_content_too_large():
    """Test that fetch_url_content raises ValueError for large content."""
    mock_response = MagicMock()
    # Use a value larger than the default Config.MAX_FILE_SIZE (10MB)
    mock_response.getheader.return_value = str(11 * 1024 * 1024)
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Content too large"):
            fetch_url_content("http://example.com/large.sh")

def test_scan_files_with_url_target(mock_tf_env, monkeypatch):
    """Test that URLs in scan_targets are fetched and scanned."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [])
    url = "https://example.com/malicious.py"
    mock_content = b"print('malicious')"

    mock_response = MagicMock()
    mock_response.getheader.return_value = str(len(mock_content))
    mock_response.read.return_value = mock_content
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        events = list(scan_files(
            scan_targets=[url],
            deep_scan=False,
            show_all=True,
            use_gpt=False
        ))

    # Should have progress events for fetching and collecting
    # Should have a result event for the fetched snippet
    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    path, own_conf, admin, user, gpt, snippet, line = results[0]
    assert path == f"[URL] {url}"
    assert own_conf == "50%"
    assert "print('malicious')" in snippet

def test_scan_files_url_fetch_error(mock_tf_env, monkeypatch):
    """Test handling of URL fetch errors during scan."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [])
    url = "https://example.com/missing.sh"

    with patch("urllib.request.urlopen", side_effect=Exception("404 Not Found")):
        events = list(scan_files(
            scan_targets=[url],
            deep_scan=False,
            show_all=True,
            use_gpt=False
        ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    path, own_conf, admin, user, gpt, snippet, line = results[0]
    assert path == url
    assert own_conf == "Fetch Error"
    assert "Could not download script" in snippet

def test_cli_url_integration(monkeypatch):
    """Test that a URL argument in CLI is correctly passed to run_cli as a target."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    url = "https://example.com/script.sh"
    test_args = ["gptscan.py", url, "--cli"]

    with patch("sys.argv", test_args):
        gptscan.main()

    mock_run_cli.assert_called_once()
    args, _ = mock_run_cli.call_args
    # First arg of run_cli is 'targets'
    assert url in args[0]
