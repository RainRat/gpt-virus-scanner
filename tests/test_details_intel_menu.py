import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json

@pytest.fixture
def mock_view_details_env(monkeypatch):
    """Setup a mock environment for view_details tests."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Setup mock_tree.item to handle both single and double argument calls
    mock_tree._item_values = {}
    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, [])
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item_func
    mock_tree.exists.side_effect = lambda iid: iid in mock_tree._item_values

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock Entry
    class MockEntry:
        def __init__(self, *args, **kwargs):
            self.val = ""
        def delete(self, start, end): self.val = ""
        def insert(self, idx, val): self.val = val
        def get(self): return self.val
        def config(self, **kwargs): pass
        def grid(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def cget(self, key): return ""

    monkeypatch.setattr(gptscan.ttk, 'Entry', MockEntry)

    # Mock tk.Label/ttk.Label
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

    # Mock ScrolledText
    class MockScrolledText:
        def __init__(self, *args, **kwargs):
            self.content = ""
            self.tags = []
        def delete(self, start, end): self.content = ""
        def insert(self, idx, val): self.content += val
        def get(self, start, end): return self.content
        def config(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def pack_forget(self, **kwargs): pass
        def tag_add(self, tag, start, end): self.tags.append((tag, start, end))
        def tag_configure(self, *args, **kwargs): pass
        def see(self, *args): pass
        def winfo_viewable(self): return True

    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MockScrolledText)

    # Mock messagebox
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Capture components from gptscan.view_details
    captured = {
        'labels': [],
        'scrolledtexts': [],
        'menus': []
    }

    def mock_label_init(*args, **kwargs):
        lbl = MockLabel(*args, **kwargs)
        captured['labels'].append(lbl)
        return lbl
    monkeypatch.setattr(gptscan.tk, 'Label', mock_label_init)
    monkeypatch.setattr(gptscan.ttk, 'Label', mock_label_init)

    def mock_st_init(*args, **kwargs):
        st = MockScrolledText(*args, **kwargs)
        captured['scrolledtexts'].append(st)
        return st
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', mock_st_init)

    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"btn_{text}"] = (btn, kwargs.get('command'))
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    class MockMenu:
        def __init__(self, master=None, **kwargs):
            self.master = master
            self.items = {}
            self.configs = {}
            captured['menus'].append(self)
        def add_command(self, **kwargs):
            label = kwargs.get('label')
            if label:
                self.items[label] = kwargs.get('command')
                captured[f"menu_{label}"] = kwargs.get('command')
        def add_separator(self): pass
        def add_cascade(self, **kwargs): pass
        def entryconfig(self, index, **kwargs):
            self.configs[index] = kwargs
        def entrycget(self, index, option):
            return self.configs.get(index, {}).get(option, "normal")

    monkeypatch.setattr(gptscan.tk, 'Menu', MockMenu)

    def mock_menubutton_init(master, **kwargs):
        mb = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"mb_{text}"] = mb
        return mb
    monkeypatch.setattr(gptscan.ttk, 'Menubutton', mock_menubutton_init)

    def mock_labelframe_init(master, **kwargs):
        lf = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"lf_{text}"] = lf
        return lf
    monkeypatch.setattr(gptscan.ttk, 'LabelFrame', mock_labelframe_init)

    return captured, mock_msgbox, mock_tree, mock_toplevel

def setup_details(captured_env, item_id, path, own_conf="90%", admin="Admin", user="User", gpt_conf="80%", snippet="snippet", line=1):
    captured, mock_msgbox, mock_tree, mock_toplevel = captured_env
    raw_vals = [path, own_conf, admin, user, gpt_conf, snippet, line]
    mock_tree._item_values[item_id] = [path, own_conf, admin, user, gpt_conf, snippet, line, json.dumps(raw_vals)]
    mock_tree.get_children.return_value = list(mock_tree._item_values.keys())
    mock_tree.selection.return_value = [item_id]
    gptscan.view_details(item_id=item_id)

def test_view_details_intel_menu_creation(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py")

    assert "mb_Intel" in captured
    assert "menu_Check on VirusTotal" in captured
    assert "menu_View Online" in captured

def test_view_details_intel_menu_state_management(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env

    # Normal file
    setup_details(mock_view_details_env, "item1", "test.py")

    intel_menu_normal = None
    for m in reversed(captured['menus']):
        if "View Online" in m.items:
            intel_menu_normal = m
            break

    assert intel_menu_normal is not None
    assert intel_menu_normal.entrycget("View Online", "state") == "normal"

    # Virtual path
    setup_details(mock_view_details_env, "item2", "[Archive] test.py")

    intel_menu_virtual = None
    for m in reversed(captured['menus']):
        if "View Online" in m.items:
            intel_menu_virtual = m
            break

    assert intel_menu_virtual is not None
    assert intel_menu_virtual.entrycget("View Online", "state") == "disabled"
