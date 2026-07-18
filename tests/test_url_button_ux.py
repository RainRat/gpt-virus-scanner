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

    # Mock messagebox to avoid tkinter dialogs in headless environments
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    yield mock_msgbox

    gptscan.root = None
    gptscan.textbox = None
    gptscan.scan_button = None

def test_normalize_and_validate_url_valid_schemes(mock_gui):
    """Test normalize_and_validate_url preserves valid schemes."""
    assert gptscan.normalize_and_validate_url("https://example.com") == "https://example.com"
    assert gptscan.normalize_and_validate_url("http://localhost:8000") == "http://localhost:8000"

def test_normalize_and_validate_url_no_scheme(mock_gui):
    """Test normalize_and_validate_url prepends https:// to scheme-less addresses."""
    assert gptscan.normalize_and_validate_url("github.com/user/repo") == "https://github.com/user/repo"
    assert gptscan.normalize_and_validate_url("localhost") == "https://localhost"

def test_normalize_and_validate_url_invalid_scheme(mock_gui):
    """Test normalize_and_validate_url rejects invalid schemes."""
    assert gptscan.normalize_and_validate_url("file:///etc/passwd") is None
    mock_gui.showerror.assert_called_with(
        "Unsupported Protocol",
        "Unsupported protocol 'file://'. Only 'http://' and 'https://' are supported."
    )

def test_normalize_and_validate_url_completely_invalid(mock_gui):
    """Test normalize_and_validate_url rejects non-web-address inputs."""
    assert gptscan.normalize_and_validate_url("invalid_input_without_dots") is None
    mock_gui.showerror.assert_called_with(
        "Invalid Web Link",
        "The entered string 'invalid_input_without_dots' does not appear to be a valid web link (http/https)."
    )

def test_normalize_and_validate_url_empty(mock_gui):
    """Test normalize_and_validate_url returns None for empty or whitespace inputs."""
    assert gptscan.normalize_and_validate_url("") is None
    assert gptscan.normalize_and_validate_url("   ") is None
    mock_gui.showerror.assert_not_called()

def test_select_url_click_updates_textbox(mock_gui, monkeypatch):
    """Test that select_url_click prompts for a URL and updates the textbox."""
    test_url = "https://example.com/malicious.sh"
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    with patch("gptscan.simpledialog.askstring", return_value=test_url) as mock_ask:
        gptscan.select_url_click()

        mock_ask.assert_called_once_with("Scan Web Link", "Enter a script web link to scan (http/https):")
        assert gptscan.textbox.get() == test_url
        mock_button_click.assert_called_once()

def test_select_url_click_cancel(mock_gui, monkeypatch):
    """Test that select_url_click does nothing if cancelled."""
    gptscan.textbox.insert(0, "previous_value")
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    with patch("gptscan.simpledialog.askstring", return_value=None):
        gptscan.select_url_click()

        assert gptscan.textbox.get() == "previous_value"
        mock_button_click.assert_not_called()

def test_select_url_click_strips_and_normalizes(mock_gui, monkeypatch):
    """Test that select_url_click strips and normalizes entered URLs."""
    test_url = "  github.com/user/repo  "
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    with patch("gptscan.simpledialog.askstring", return_value=test_url):
        gptscan.select_url_click()

        assert gptscan.textbox.get() == "https://github.com/user/repo"
        mock_button_click.assert_called_once()
