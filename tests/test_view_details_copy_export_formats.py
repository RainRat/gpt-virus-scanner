import pytest
from unittest.mock import MagicMock
import gptscan
import json

@pytest.fixture
def mock_details_copy_env(monkeypatch):
    """Setup environment for testing view_details copy commands."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line") if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    raw_vals = ["test.py", "90%", "Admin notes", "User notes", "80%", "print('hello')", 1]
    mock_tree._item_values = {"item1": ["test.py", "90%", "Admin notes", "User notes", "80%", "print('hello')", 1, json.dumps(raw_vals)]}
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]
    mock_tree.exists.side_effect = lambda iid: iid in mock_tree._item_values

    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, [])
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item_func

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Captured menu entries
    captured_commands = {}

    class MockLabel:
        def __init__(self, *args, **kwargs):
            self.config_data = {}
        def config(self, **kwargs): self.config_data.update(kwargs)
        def cget(self, key): return self.config_data.get(key, "")
        def grid(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def grid_forget(self): pass
        def pack_forget(self): pass
        def winfo_viewable(self): return True

    monkeypatch.setattr(gptscan.tk, 'Label', MockLabel)
    monkeypatch.setattr(gptscan.ttk, 'Label', MockLabel)

    class MockMenu:
        def __init__(self, master=None, **kwargs): pass
        def add_command(self, **kwargs):
            label = kwargs.get('label')
            cmd = kwargs.get('command')
            if label and cmd:
                captured_commands[label] = cmd
        def add_separator(self): pass
        def add_cascade(self, **kwargs): pass
        def entryconfig(self, index, **kwargs): pass

    monkeypatch.setattr(gptscan.tk, 'Menu', MockMenu)

    gptscan.view_details(item_id="item1")
    return captured_commands, mock_root

def test_view_details_copy_as_junit_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as JUnit XML" in captured_commands
    cmd = captured_commands["Copy as JUnit XML"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "<testsuite" in copied_content or "<testcase" in copied_content

def test_view_details_copy_as_csv_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as CSV" in captured_commands
    cmd = captured_commands["Copy as CSV"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet" in copied_content
    assert "test.py" in copied_content

def test_view_details_copy_as_tsv_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as TSV" in captured_commands
    cmd = captured_commands["Copy as TSV"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "path\tline\town_conf\tgpt_conf\tadmin_desc\tend-user_desc\tsnippet" in copied_content
    assert "test.py" in copied_content

def test_view_details_copy_as_html_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as HTML" in captured_commands
    cmd = captured_commands["Copy as HTML"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "<html" in copied_content.lower() or "<table" in copied_content.lower() or "test.py" in copied_content

def test_view_details_copy_as_json_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as JSON" in captured_commands
    cmd = captured_commands["Copy as JSON"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    parsed = json.loads(copied_content)
    assert parsed["path"] == "test.py"

def test_view_details_copy_as_markdown_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as Markdown" in captured_commands
    cmd = captured_commands["Copy as Markdown"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "test.py" in copied_content

def test_view_details_copy_as_sarif_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as SARIF" in captured_commands
    cmd = captured_commands["Copy as SARIF"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    parsed = json.loads(copied_content)
    assert "$schema" in parsed or "version" in parsed

def test_view_details_copy_as_report_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as Triage Report" in captured_commands
    cmd = captured_commands["Copy as Triage Report"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "test.py" in copied_content

def test_view_details_copy_as_xml_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as XML" in captured_commands
    cmd = captured_commands["Copy as XML"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "<" in copied_content and "test.py" in copied_content

def test_view_details_copy_as_yaml_details(mock_details_copy_env):
    captured_commands, mock_root = mock_details_copy_env
    assert "Copy as YAML" in captured_commands
    cmd = captured_commands["Copy as YAML"]

    mock_root.clipboard_clear.reset_mock()
    mock_root.clipboard_append.reset_mock()

    cmd()

    mock_root.clipboard_clear.assert_called_once()
    assert mock_root.clipboard_append.called
    copied_content = mock_root.clipboard_append.call_args[0][0]
    assert "test.py" in copied_content
