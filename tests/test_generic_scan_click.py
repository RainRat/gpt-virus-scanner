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


def test_scan_ssh_config_click(mocker):
    mocker.patch("gptscan.get_ssh_config_paths", return_value=["/ssh"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_ssh_config_click()
    mock_generic.assert_called_once_with(
        gptscan.get_ssh_config_paths,
        "SSH Configuration",
        "No SSH configuration or authorized_keys files were found to scan.",
        "SSH Configuration Error"
    )


def test_scan_network_config_click(mocker):
    mocker.patch("gptscan.get_network_config_paths", return_value=["/network"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_network_config_click()
    mock_generic.assert_called_once_with(
        gptscan.get_network_config_paths,
        "Network Configuration",
        "No network configuration files were found to scan.",
        "Network Configuration Error"
    )


def test_scan_python_packages_click(mocker):
    mocker.patch("gptscan.get_python_package_paths", return_value=["/python"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_python_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_python_package_paths,
        "Python Packages",
        "No Python site-packages folders were found to scan.",
        "Python Packages Error"
    )


def test_scan_env_files_click(mocker):
    mocker.patch("gptscan.get_env_file_paths", return_value=[".env"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_env_files_click()
    mock_generic.assert_called_once_with(
        gptscan.get_env_file_paths,
        "Env Files",
        "No common .env files were found.",
        "Env Files Error"
    )


def test_scan_nodejs_packages_click(mocker):
    mocker.patch("gptscan.get_nodejs_package_paths", return_value=["/node"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_nodejs_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_nodejs_package_paths,
        "Node.js Packages",
        "No global Node.js package folders were found to scan.",
        "Node.js Packages Error"
    )


def test_scan_browser_bookmarks_click(mocker):
    mocker.patch("gptscan.get_browser_bookmarks_snippets", return_value=[("bm", b"js")])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_browser_bookmarks_click()
    mock_generic.assert_called_once_with(
        gptscan.get_browser_bookmarks_snippets,
        "Browser Bookmarks",
        "No suspicious browser bookmarklets (javascript: or data: URLs) were found.",
        "Browser Bookmarks Error",
        is_snippets=True
    )


def test_scan_browser_extensions_click(mocker):
    mocker.patch("gptscan.get_browser_extensions_paths", return_value=["/extensions"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_browser_extensions_click()
    mock_generic.assert_called_once_with(
        gptscan.get_browser_extensions_paths,
        "Browser Extensions",
        "No browser extension folders were found to scan.",
        "Browser Extensions Error"
    )


def test_scan_editor_extensions_click(mocker):
    mocker.patch("gptscan.get_editor_extensions_paths", return_value=["/editor"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_editor_extensions_click()
    mock_generic.assert_called_once_with(
        gptscan.get_editor_extensions_paths,
        "Editor Extensions",
        "No editor extension folders were found to scan.",
        "Editor Extensions Error"
    )


def test_scan_desktop_click(mocker):
    mocker.patch("gptscan.get_desktop_paths", return_value=["/desktop"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_desktop_click()
    mock_generic.assert_called_once_with(
        gptscan.get_desktop_paths,
        "Desktop",
        "The Desktop folder was not found on this system.",
        "Desktop Error"
    )


def test_scan_temp_click(mocker):
    mocker.patch("gptscan.get_temp_paths", return_value=["/temp"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_temp_click()
    mock_generic.assert_called_once_with(
        gptscan.get_temp_paths,
        "Temporary Folders",
        "No common temporary folders were found on this system.",
        "Temporary Folders Error"
    )


def test_scan_ruby_gems_click(mocker):
    mocker.patch("gptscan.get_ruby_gems_paths", return_value=["/gems"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_ruby_gems_click()
    mock_generic.assert_called_once_with(
        gptscan.get_ruby_gems_paths,
        "Ruby Gems",
        "No Ruby gems folders were found to scan.",
        "Ruby Gems Error"
    )


def test_scan_php_packages_click(mocker):
    mocker.patch("gptscan.get_php_packages_paths", return_value=["/php"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_php_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_php_packages_paths,
        "PHP Packages",
        "No global PHP package folders were found to scan.",
        "PHP Packages Error"
    )


def test_scan_rust_packages_click(mocker):
    mocker.patch("gptscan.get_rust_packages_paths", return_value=["/rust"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_rust_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_rust_packages_paths,
        "Rust Packages",
        "No global Rust package folders were found to scan.",
        "Rust Packages Error"
    )


def test_scan_go_packages_click(mocker):
    mocker.patch("gptscan.get_go_packages_paths", return_value=["/go"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_go_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_go_packages_paths,
        "Go Packages",
        "No Go package folders were found to scan.",
        "Go Packages Error"
    )


def test_scan_java_packages_click(mocker):
    mocker.patch("gptscan.get_java_packages_paths", return_value=["/java"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_java_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_java_packages_paths,
        "Java Packages",
        "No Java package folders were found to scan.",
        "Java Packages Error"
    )


def test_scan_dotnet_packages_click(mocker):
    mocker.patch("gptscan.get_dotnet_packages_paths", return_value=["/dotnet"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_dotnet_packages_click()
    mock_generic.assert_called_once_with(
        gptscan.get_dotnet_packages_paths,
        ".NET Packages",
        "No .NET NuGet package folders were found to scan.",
        ".NET Packages Error"
    )


def test_scan_documents_click(mocker):
    mocker.patch("gptscan.get_documents_paths", return_value=["/docs"])
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    gptscan.scan_documents_click()
    mock_generic.assert_called_once_with(
        gptscan.get_documents_paths,
        "Documents",
        "The standard Documents folder was not found on this system.",
        "Documents Error"
    )
