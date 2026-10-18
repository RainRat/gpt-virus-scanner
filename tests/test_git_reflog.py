import pytest
import subprocess
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import get_git_reflog_snippets, scan_git_reflog_click

def test_get_git_reflog_snippets_no_git():
    with patch('gptscan._get_git_info', return_value=(None, None)):
        assert get_git_reflog_snippets() == []

def test_get_git_reflog_snippets_success(monkeypatch):
    monkeypatch.setattr('gptscan._get_git_info', lambda *a, **kw: ('/fake/root', 'rel/path'))

    # Mock git reflog output
    reflog_output = "abc1234 HEAD@{0}: commit: first commit\ndef5678 HEAD@{1}: commit: second commit"
    monkeypatch.setattr('subprocess.check_output', MagicMock(side_effect=[
        reflog_output, # for git reflog
        "diff content 1", # for git show abc1234
        "diff content 2"  # for git show def5678
    ]))

    snippets = get_git_reflog_snippets(count=2)
    assert len(snippets) == 2
    assert snippets[0][0] == "[Git Reflog] abc1234 HEAD@{0}: commit: first commit"
    assert snippets[0][1] == b"diff content 1"
    assert snippets[1][0] == "[Git Reflog] def5678 HEAD@{1}: commit: second commit"
    assert snippets[1][1] == b"diff content 2"

def test_get_git_reflog_snippets_partial_failure_and_empty_lines(monkeypatch):
    monkeypatch.setattr('gptscan._get_git_info', lambda *a, **kw: ('/fake/root', 'rel/path'))

    # Reflog output containing an empty line, a single-token line (no subject), and standard lines
    reflog_output = "\n  \nabc1234 HEAD@{0}: commit: first\nfailed123\ndef5678\n"

    def mock_check_output(cmd, **kwargs):
        if cmd[1] == "reflog":
            return reflog_output
        elif cmd[1] == "show":
            commit = cmd[3]
            if commit == "failed123":
                raise subprocess.CalledProcessError(1, cmd)
            elif commit == "abc1234":
                return "diff for abc1234"
            elif commit == "def5678":
                return "diff for def5678"
        return ""

    monkeypatch.setattr('subprocess.check_output', mock_check_output)

    snippets = get_git_reflog_snippets(count=5)
    assert len(snippets) == 2
    assert snippets[0][0] == "[Git Reflog] abc1234 HEAD@{0}: commit: first"
    assert snippets[0][1] == b"diff for abc1234"
    assert snippets[1][0] == "[Git Reflog] def5678 "
    assert snippets[1][1] == b"diff for def5678"

def test_get_git_reflog_snippets_cmd_exception(monkeypatch):
    monkeypatch.setattr('gptscan._get_git_info', lambda *a, **kw: ('/fake/root', 'rel/path'))
    monkeypatch.setattr('subprocess.check_output', MagicMock(side_effect=subprocess.CalledProcessError(1, ['git', 'reflog'])))

    assert get_git_reflog_snippets() == []

def test_scan_git_reflog_click_cancel(monkeypatch):
    monkeypatch.setattr('tkinter.simpledialog.askinteger', lambda *a, **kw: None)
    mock_button_click = MagicMock()
    monkeypatch.setattr('gptscan.button_click', mock_button_click)

    scan_git_reflog_click()
    mock_button_click.assert_not_called()

def test_scan_git_reflog_click_success(monkeypatch):
    monkeypatch.setattr('tkinter.simpledialog.askinteger', lambda *a, **kw: 3)

    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""
    with patch('gptscan.textbox', mock_textbox):
        mock_get_snippets = MagicMock(return_value=[("name", b"content")])
        mock_button_click = MagicMock()
        monkeypatch.setattr('gptscan.get_git_reflog_snippets', mock_get_snippets)
        monkeypatch.setattr('gptscan.button_click', mock_button_click)

        scan_git_reflog_click()

        mock_get_snippets.assert_called_once_with(".", count=3)
        mock_button_click.assert_called_once_with(extra_snippets=[("name", b"content")])

def test_scan_git_reflog_click_with_count(monkeypatch):
    monkeypatch.setattr('gptscan.simpledialog.askinteger', MagicMock())
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    with patch('gptscan.textbox', mock_textbox):
        mock_get_snippets = MagicMock(return_value=[("name", b"content")])
        mock_button_click = MagicMock()
        monkeypatch.setattr('gptscan.get_git_reflog_snippets', mock_get_snippets)
        monkeypatch.setattr('gptscan.button_click', mock_button_click)

        scan_git_reflog_click(count=15)

        mock_get_snippets.assert_called_once_with(".", count=15)
        mock_button_click.assert_called_once_with(extra_snippets=[("name", b"content")])

def test_scan_git_reflog_click_empty_results(monkeypatch):
    monkeypatch.setattr('tkinter.simpledialog.askinteger', lambda *a, **kw: 5)
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    with patch('gptscan.textbox', mock_textbox):
        monkeypatch.setattr('gptscan.get_git_reflog_snippets', lambda *a, **kw: [])
        mock_info = MagicMock()
        monkeypatch.setattr('gptscan.messagebox.showinfo', mock_info)
        mock_button_click = MagicMock()
        monkeypatch.setattr('gptscan.button_click', mock_button_click)

        scan_git_reflog_click()

        mock_button_click.assert_not_called()
        mock_info.assert_called_once()

def test_scan_git_reflog_click_exception(monkeypatch):
    monkeypatch.setattr('tkinter.simpledialog.askinteger', lambda *a, **kw: 5)
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    with patch('gptscan.textbox', mock_textbox):
        monkeypatch.setattr('gptscan.get_git_reflog_snippets', MagicMock(side_effect=RuntimeError("git reflog failed")))
        mock_warn = MagicMock()
        monkeypatch.setattr('gptscan.messagebox.showwarning', mock_warn)

        scan_git_reflog_click()

        mock_warn.assert_called_once()
