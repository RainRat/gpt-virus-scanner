import json
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details

def test_copy_sha256_details_success(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="code snippet")

    monkeypatch.setattr(gptscan, "get_effective_sha256", lambda path, snippet: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    gptscan.view_details(item_id="item1")

    assert "menu_Copy SHA256" in captured
    copy_sha256_cmd = captured["menu_Copy SHA256"]

    from gptscan import root as mock_root
    copy_sha256_cmd()

    mock_root.clipboard_clear.assert_called()
    mock_root.clipboard_append.assert_called_with("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

def test_copy_sha256_details_failure(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="code snippet")

    monkeypatch.setattr(gptscan, "get_effective_sha256", lambda path, snippet: None)

    gptscan.view_details(item_id="item1")

    assert "menu_Copy SHA256" in captured
    copy_sha256_cmd = captured["menu_Copy SHA256"]

    copy_sha256_cmd()

    mock_msgbox.showwarning.assert_called_once_with(
        "Error", "Could not calculate file hash.", parent=mock_toplevel
    )
