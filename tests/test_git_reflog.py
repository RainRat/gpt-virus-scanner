import pytest
import subprocess
from unittest.mock import MagicMock, patch
from gptscan import get_git_reflog_snippets, scan_git_reflog_click

def test_get_git_reflog_snippets_no_git():
    with patch('gptscan._get_git_info', return_value=(None, None)):
        assert get_git_reflog_snippets() == []

def test_get_git_reflog_snippets_success(mocker):
    mocker.patch('gptscan._get_git_info', return_value=('/fake/root', 'rel/path'))

    # Mock git reflog output
    reflog_output = "abc1234 HEAD@{0}: commit: first commit\ndef5678 HEAD@{1}: commit: second commit"
    mocker.patch('subprocess.check_output', side_effect=[
        reflog_output, # for git reflog
        "diff content 1", # for git show abc1234
        "diff content 2"  # for git show def5678
    ])

    snippets = get_git_reflog_snippets(count=2)
    assert len(snippets) == 2
    assert snippets[0][0] == "[Git Reflog] abc1234 HEAD@{0}: commit: first commit"
    assert snippets[0][1] == b"diff content 1"
    assert snippets[1][0] == "[Git Reflog] def5678 HEAD@{1}: commit: second commit"
    assert snippets[1][1] == b"diff content 2"

def test_scan_git_reflog_click_cancel(mocker):
    # Mock simpledialog to return None (cancel)
    mocker.patch('tkinter.simpledialog.askinteger', return_value=None)
    mock_button_click = mocker.patch('gptscan.button_click')

    scan_git_reflog_click()
    mock_button_click.assert_not_called()

def test_scan_git_reflog_click_success(mocker):
    mocker.patch('tkinter.simpledialog.askinteger', return_value=3)

    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""
    with patch('gptscan.textbox', mock_textbox):
        mock_get_snippets = mocker.patch('gptscan.get_git_reflog_snippets', return_value=[("name", b"content")])
        mock_button_click = mocker.patch('gptscan.button_click')

        scan_git_reflog_click()

        mock_get_snippets.assert_called_once_with(".", count=3)
        mock_button_click.assert_called_once_with(extra_snippets=[("name", b"content")])
