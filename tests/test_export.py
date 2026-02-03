import csv
import pytest
from unittest.mock import MagicMock
import tkinter.filedialog
import gptscan

def test_export_results_cancels(monkeypatch):
    """Verify that cancelling the file dialog aborts the export process."""
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', MagicMock(return_value=''))

    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_open = MagicMock()
    monkeypatch.setattr(gptscan, 'open', mock_open, raising=False)

    gptscan.export_results()

    mock_open.assert_not_called()

def test_export_results_success(monkeypatch, tmp_path):
    """Verify that data from the treeview is correctly written to a CSV file."""
    file_path = tmp_path / "export.csv"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', MagicMock(return_value=str(file_path)))

    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("Col1", "Col2")
    mock_tree.get_children.return_value = ("item1", "item2")

    def get_item(iid):
        if iid == "item1":
            return {"values": ("val1a", "val1b")}
        return {"values": ("val2a", "val2b")}
    mock_tree.item.side_effect = get_item

    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    gptscan.export_results()

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "Col1,Col2"
    assert lines[1] == "val1a,val1b"
    assert lines[2] == "val2a,val2b"

def test_export_results_handles_error(monkeypatch):
    """Verify that file I/O errors are caught and displayed to the user."""
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', MagicMock(return_value="out.csv"))

    def fail_open(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(gptscan, "open", fail_open, raising=False)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    gptscan.export_results()

    mock_msgbox.showerror.assert_called_once()
    args, _ = mock_msgbox.showerror.call_args
    assert args[0] == "Export Failed"
    assert "Disk full" in args[1]
