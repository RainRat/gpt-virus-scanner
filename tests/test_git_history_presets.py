import pytest
from unittest.mock import MagicMock, patch

# Mock tkinter before importing gptscan
import sys
import tkinter as tk
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.simpledialog'] = MagicMock()

import gptscan

# Mock the global textbox object
gptscan.textbox = MagicMock()

def test_scan_git_history_click_with_count():
    with patch('gptscan.simpledialog.askinteger') as mock_ask:
        with patch('gptscan.get_git_history_snippets', return_value=['snippet1']) as mock_get:
            with patch('gptscan.button_click') as mock_click:
                gptscan.textbox.get.return_value = '.'
                gptscan.scan_git_history_click(count=10)
                mock_ask.assert_not_called()
                mock_get.assert_called_with('.', count=10)
                mock_click.assert_called_once()

def test_scan_git_history_click_without_count():
    with patch('gptscan.simpledialog.askinteger', return_value=5) as mock_ask:
        with patch('gptscan.get_git_history_snippets', return_value=['snippet1']) as mock_get:
            with patch('gptscan.button_click') as mock_click:
                gptscan.textbox.get.return_value = '.'
                gptscan.scan_git_history_click()
                mock_ask.assert_called_once()
                mock_get.assert_called_with('.', count=5)
                mock_click.assert_called_once()

def test_scan_git_reflog_click_with_count():
    with patch('gptscan.simpledialog.askinteger') as mock_ask:
        with patch('gptscan.get_git_reflog_snippets', return_value=['snippet1']) as mock_get:
            with patch('gptscan.button_click') as mock_click:
                gptscan.textbox.get.return_value = '.'
                gptscan.scan_git_reflog_click(count=15)
                mock_ask.assert_not_called()
                mock_get.assert_called_with('.', count=15)
                mock_click.assert_called_once()
