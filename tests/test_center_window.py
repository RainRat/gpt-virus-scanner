import pytest
import tkinter as tk
from unittest.mock import MagicMock
import gptscan


def test_center_window_valid(monkeypatch):
    mock_window = MagicMock()
    mock_window.winfo_width.return_value = 200
    mock_window.winfo_height.return_value = 100

    mock_parent = MagicMock()
    mock_parent.winfo_width.return_value = 800
    mock_parent.winfo_height.return_value = 600
    mock_parent.winfo_x.return_value = 100
    mock_parent.winfo_y.return_value = 50

    gptscan.center_window(mock_window, mock_parent)

    mock_window.update_idletasks.assert_called_once()
    mock_window.geometry.assert_called_once_with("200x100+400+300")


def test_center_window_small_dimensions_fallback_geometry():
    mock_window = MagicMock()
    mock_window.winfo_width.return_value = 1
    mock_window.winfo_height.return_value = 1
    mock_window.geometry.return_value = "300x150+0+0"

    mock_parent = MagicMock()
    mock_parent.winfo_width.return_value = 1000
    mock_parent.winfo_height.return_value = 800
    mock_parent.winfo_x.return_value = 0
    mock_parent.winfo_y.return_value = 0

    gptscan.center_window(mock_window, mock_parent)

    mock_window.geometry.assert_called_with("300x150+350+325")


def test_center_window_tcl_error():
    mock_window = MagicMock()
    mock_window.update_idletasks.side_effect = tk.TclError("Window destroyed")
    mock_parent = MagicMock()

    # Should not raise exception
    gptscan.center_window(mock_window, mock_parent)


def test_center_window_value_error():
    mock_window = MagicMock()
    mock_window.winfo_width.return_value = "invalid"
    mock_parent = MagicMock()

    # Should not raise exception
    gptscan.center_window(mock_window, mock_parent)


def test_center_window_attribute_error():
    mock_window = None
    mock_parent = MagicMock()

    # Should not raise exception
    gptscan.center_window(mock_window, mock_parent)


def test_center_window_type_error():
    mock_window = MagicMock()
    mock_window.winfo_width.return_value = [100]
    mock_parent = MagicMock()

    # Should not raise exception
    gptscan.center_window(mock_window, mock_parent)
