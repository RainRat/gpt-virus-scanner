import pytest
from unittest.mock import MagicMock, patch
import gptscan
from typing import List, Tuple

def test_generic_scan_click_paths_success(mocker):
    mock_get_data = MagicMock(return_value=["/path/1", "/path/2"])
    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")
    mock_showinfo = mocker.patch("tkinter.messagebox.showinfo")
    mock_showwarning = mocker.patch("tkinter.messagebox.showwarning")

    gptscan._generic_scan_click(mock_get_data, "Test Title", "No data", "Error Title")

    mock_get_data.assert_called_once()
    mock_set_target.assert_called_once_with(["/path/1", "/path/2"])
    mock_button_click.assert_called_once_with()
    mock_showinfo.assert_not_called()
    mock_showwarning.assert_not_called()

def test_generic_scan_click_snippets_success(mocker):
    snippets = [("snippet1", b"content1"), ("snippet2", b"content2")]
    mock_get_data = MagicMock(return_value=snippets)
    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")
    mock_showinfo = mocker.patch("tkinter.messagebox.showinfo")
    mock_showwarning = mocker.patch("tkinter.messagebox.showwarning")

    gptscan._generic_scan_click(mock_get_data, "Test Title", "No data", "Error Title", is_snippets=True)

    mock_get_data.assert_called_once()
    mock_set_target.assert_not_called()
    mock_button_click.assert_called_once_with(extra_snippets=snippets)
    mock_showinfo.assert_not_called()
    mock_showwarning.assert_not_called()

def test_generic_scan_click_no_data(mocker):
    mock_get_data = MagicMock(return_value=[])
    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")
    mock_showinfo = mocker.patch("tkinter.messagebox.showinfo")
    mock_showwarning = mocker.patch("tkinter.messagebox.showwarning")

    gptscan._generic_scan_click(mock_get_data, "Test Title", "Failure Message", "Error Title")

    mock_get_data.assert_called_once()
    mock_set_target.assert_not_called()
    mock_button_click.assert_not_called()
    mock_showinfo.assert_called_once_with("Test Title", "Failure Message")
    mock_showwarning.assert_not_called()

def test_generic_scan_click_exception(mocker):
    mock_get_data = MagicMock(side_effect=Exception("Test Error"))
    mock_set_target = mocker.patch("gptscan._set_scan_target")
    mock_button_click = mocker.patch("gptscan.button_click")
    mock_showinfo = mocker.patch("tkinter.messagebox.showinfo")
    mock_showwarning = mocker.patch("tkinter.messagebox.showwarning")

    gptscan._generic_scan_click(mock_get_data, "Test Title", "No data", "Error Title")

    mock_get_data.assert_called_once()
    mock_set_target.assert_not_called()
    mock_button_click.assert_not_called()
    mock_showinfo.assert_not_called()
    # Check that it includes the error title and mentions "test title" in lowercase
    args, kwargs = mock_showwarning.call_args
    assert args[0] == "Error Title"
    assert "test title" in args[1]
    assert "Test Error" in args[1]

def test_refactored_scan_downloads_click(mocker):
    """Verify one of the refactored functions to ensure it correctly calls the helper."""
    mock_get_paths = mocker.patch("gptscan.get_downloads_paths", return_value=["/downloads"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")

    gptscan.scan_downloads_click()

    mock_generic.assert_called_once_with(
        gptscan.get_downloads_paths,
        "Downloads",
        "The standard Downloads folder was not found on this system.",
        "Downloads Error"
    )

def test_refactored_scan_env_vars_click(mocker):
    """Verify a snippet-based refactored function."""
    mock_get_snippets = mocker.patch("gptscan.get_environment_variable_snippets", return_value=[("env", b"val")])
    mock_generic = mocker.patch("gptscan._generic_scan_click")

    gptscan.scan_env_vars_click()

    mock_generic.assert_called_once_with(
        gptscan.get_environment_variable_snippets,
        "Environment Variables",
        "No non-empty environment variables were found.",
        "Environment Variables Error",
        is_snippets=True
    )

def test_refactored_scan_network_config_click(mocker):
    """Verify that scan_network_config_click correctly calls the _generic_scan_click helper."""
    mock_get_paths = mocker.patch("gptscan.get_network_config_paths", return_value=["/etc/hosts"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")

    gptscan.scan_network_config_click()

    mock_generic.assert_called_once_with(
        gptscan.get_network_config_paths,
        "Network Configuration",
        "No network configuration files were found to scan.",
        "Network Configuration Error"
    )
