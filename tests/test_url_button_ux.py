from unittest.mock import MagicMock, patch
import pytest
import gptscan

@pytest.fixture
def mock_gui(monkeypatch):
    """Setup a mock GUI environment for testing button clicks."""
    # Reset textbox value for each test
    current_text = [""]

    mock_textbox = MagicMock()
    mock_textbox.get.side_effect = lambda: current_text[0]
    def mock_insert(idx, val):
        current_text[0] = val
    def mock_delete(start, end):
        current_text[0] = ""
    mock_textbox.insert.side_effect = mock_insert
    mock_textbox.delete.side_effect = mock_delete

    gptscan.root = MagicMock()
    gptscan.textbox = mock_textbox
    gptscan.scan_button = MagicMock()

    yield

    gptscan.root = None
    gptscan.textbox = None
    gptscan.scan_button = None

def test_select_url_click_updates_textbox(mock_gui):
    """Test that select_url_click prompts for a URL and updates the textbox."""
    test_url = "https://example.com/malicious.sh"

    with patch("gptscan.simpledialog.askstring", return_value=test_url) as mock_ask:
        gptscan.select_url_click()

        mock_ask.assert_called_once_with("Scan URL", "Enter a script URL to scan (http/https):")
        assert gptscan.textbox.get() == test_url

def test_select_url_click_cancel(mock_gui):
    """Test that select_url_click does nothing if cancelled."""
    gptscan.textbox.insert(0, "previous_value")

    with patch("gptscan.simpledialog.askstring", return_value=None):
        gptscan.select_url_click()

        assert gptscan.textbox.get() == "previous_value"

def test_select_url_click_strips_whitespace(mock_gui):
    """Test that select_url_click strips whitespace from the entered URL."""
    test_url = "  https://example.com/clean.py  "

    with patch("gptscan.simpledialog.askstring", return_value=test_url):
        gptscan.select_url_click()

        assert gptscan.textbox.get() == "https://example.com/clean.py"
