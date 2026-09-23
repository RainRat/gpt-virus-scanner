from unittest.mock import MagicMock
import pytest
import gptscan


class MockVar(MagicMock):
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self._val = value

    def get(self):
        return self._val

    def set(self, val):
        self._val = val

    def trace_add(self, *a, **kw):
        pass


@pytest.fixture(autouse=True)
def restore_config_and_gui_state():
    orig_last_path = gptscan.Config.last_path
    orig_deep_scan = gptscan.Config.deep_scan
    orig_git_changes_only = gptscan.Config.git_changes_only
    orig_show_all_files = gptscan.Config.show_all_files
    orig_scan_all_files = gptscan.Config.scan_all_files
    orig_use_ai_analysis = gptscan.Config.use_ai_analysis
    orig_provider = gptscan.Config.provider
    orig_model_name = gptscan.Config.model_name
    orig_tree = getattr(gptscan, "tree", None)
    orig_textbox = getattr(gptscan, "textbox", None)
    orig_filter_entry = getattr(gptscan, "filter_entry", None)
    orig_filter_var = getattr(gptscan, "filter_var", None)

    yield

    gptscan.Config.last_path = orig_last_path
    gptscan.Config.deep_scan = orig_deep_scan
    gptscan.Config.git_changes_only = orig_git_changes_only
    gptscan.Config.show_all_files = orig_show_all_files
    gptscan.Config.scan_all_files = orig_scan_all_files
    gptscan.Config.use_ai_analysis = orig_use_ai_analysis
    gptscan.Config.provider = orig_provider
    gptscan.Config.model_name = orig_model_name
    gptscan.tree = orig_tree
    gptscan.textbox = orig_textbox
    gptscan.filter_entry = orig_filter_entry
    gptscan.filter_var = orig_filter_var


@pytest.fixture
def mock_combo_widget():
    combo = MagicMock()
    combo.cget.return_value = "*"
    combo.get.return_value = "/scanned/directory"
    return combo


@pytest.fixture
def mock_tree_widget():
    tree = MagicMock()
    tree.tag_configure = MagicMock()
    tree.column = MagicMock()
    tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line", "orig_json")
    return tree


def test_on_close_window_lifecycle_saves_config_and_destroys_root(monkeypatch, mock_combo_widget, mock_tree_widget):
    mock_root = MagicMock()
    mock_root.after = MagicMock()
    protocol_handlers = {}

    def mock_protocol(name, func=None):
        if func:
            protocol_handlers[name] = func
        return protocol_handlers.get(name)

    mock_root.protocol.side_effect = mock_protocol

    mock_save_settings = MagicMock()
    mock_save_apikey = MagicMock()

    monkeypatch.setattr(gptscan.Config, "save_settings", mock_save_settings)
    monkeypatch.setattr(gptscan.Config, "save_apikey", mock_save_apikey)

    monkeypatch.setattr(gptscan.Config, "deep_scan", True)
    monkeypatch.setattr(gptscan.Config, "git_changes_only", False)
    monkeypatch.setattr(gptscan.Config, "show_all_files", True)
    monkeypatch.setattr(gptscan.Config, "scan_all_files", False)
    monkeypatch.setattr(gptscan.Config, "use_ai_analysis", True)
    monkeypatch.setattr(gptscan.Config, "provider", "openai")
    monkeypatch.setattr(gptscan.Config, "model_name", "gpt-4o-mini")

    monkeypatch.setattr(gptscan.tk, "BooleanVar", lambda value=False, **kw: MockVar(value))
    monkeypatch.setattr(gptscan.tk, "StringVar", lambda value="", **kw: MockVar(value))

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)
    monkeypatch.setattr(gptscan.ttk, "Combobox", lambda *a, **kw: mock_combo_widget)
    monkeypatch.setattr(gptscan.ttk, "Treeview", lambda *a, **kw: mock_tree_widget)

    gptscan.create_gui()

    assert "WM_DELETE_WINDOW" in protocol_handlers
    on_close_fn = protocol_handlers["WM_DELETE_WINDOW"]
    on_close_fn()

    assert gptscan.Config.last_path == "/scanned/directory"
    assert gptscan.Config.deep_scan is True
    assert gptscan.Config.git_changes_only is False
    assert gptscan.Config.show_all_files is True
    assert gptscan.Config.scan_all_files is False
    assert gptscan.Config.use_ai_analysis is True
    assert gptscan.Config.provider == "openai"
    assert gptscan.Config.model_name == "gpt-4o-mini"

    mock_save_settings.assert_called_once()
    mock_save_apikey.assert_called_once()
    mock_root.destroy.assert_called_once()


def test_on_tree_double_click_cell_region(monkeypatch, mock_combo_widget, mock_tree_widget):
    mock_tree = mock_tree_widget
    tree_bindings = {}
    mock_tree.bind.side_effect = lambda event, func: tree_bindings.update({event: func})

    mock_view_details = MagicMock()
    monkeypatch.setattr(gptscan, "view_details", mock_view_details)

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)
    monkeypatch.setattr(gptscan.ttk, "Combobox", lambda *a, **kw: mock_combo_widget)
    monkeypatch.setattr(gptscan.ttk, "Treeview", lambda *a, **kw: mock_tree)

    gptscan.create_gui()

    assert "<Double-1>" in tree_bindings
    double_click_fn = tree_bindings["<Double-1>"]

    cell_event = MagicMock()
    cell_event.x = 50
    cell_event.y = 50
    mock_tree.identify_region.return_value = "cell"

    double_click_fn(cell_event)
    mock_view_details.assert_called_once_with(cell_event)


def test_on_tree_double_click_non_cell_region(monkeypatch, mock_combo_widget, mock_tree_widget):
    mock_tree = mock_tree_widget
    tree_bindings = {}
    mock_tree.bind.side_effect = lambda event, func: tree_bindings.update({event: func})

    mock_view_details = MagicMock()
    monkeypatch.setattr(gptscan, "view_details", mock_view_details)

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)
    monkeypatch.setattr(gptscan.ttk, "Combobox", lambda *a, **kw: mock_combo_widget)
    monkeypatch.setattr(gptscan.ttk, "Treeview", lambda *a, **kw: mock_tree)

    gptscan.create_gui()

    double_click_fn = tree_bindings["<Double-1>"]

    heading_event = MagicMock()
    heading_event.x = 50
    heading_event.y = 10
    mock_tree.identify_region.return_value = "heading"

    double_click_fn(heading_event)
    mock_view_details.assert_not_called()


def test_clear_target_callback(monkeypatch, mock_combo_widget, mock_tree_widget):
    mock_textbox = mock_combo_widget
    mock_update_visibility = MagicMock()

    monkeypatch.setattr(gptscan, "update_clear_target_visibility", mock_update_visibility)

    button_cmds = []

    def mock_button(*args, **kwargs):
        btn = MagicMock()
        if "command" in kwargs:
            button_cmds.append((kwargs.get("text"), kwargs["command"]))
        return btn

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "Button", mock_button)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)
    monkeypatch.setattr(gptscan.ttk, "Combobox", lambda *a, **kw: mock_textbox)
    monkeypatch.setattr(gptscan.ttk, "Treeview", lambda *a, **kw: mock_tree_widget)

    gptscan.create_gui()

    clear_target_cmd = next(cmd for text, cmd in button_cmds if text == "×")
    clear_target_cmd()

    mock_textbox.delete.assert_called_once_with(0, gptscan.tk.END)
    assert mock_update_visibility.called
    assert mock_textbox.focus_set.called


def test_clear_filter_callback(monkeypatch, mock_combo_widget, mock_tree_widget):
    mock_apply_filter = MagicMock()
    mock_filter_entry = MagicMock()
    mock_filter_var = MagicMock()

    monkeypatch.setattr(gptscan, "_apply_filter", mock_apply_filter)

    button_cmds = []

    def mock_button(*args, **kwargs):
        btn = MagicMock()
        if "command" in kwargs:
            button_cmds.append((kwargs.get("text"), kwargs["command"]))
        return btn

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.tk, "StringVar", lambda value="", **kw: mock_filter_var)
    monkeypatch.setattr(gptscan.ttk, "Button", mock_button)
    monkeypatch.setattr(gptscan.ttk, "Entry", lambda *a, **kw: mock_filter_entry)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)
    monkeypatch.setattr(gptscan.ttk, "Combobox", lambda *a, **kw: mock_combo_widget)
    monkeypatch.setattr(gptscan.ttk, "Treeview", lambda *a, **kw: mock_tree_widget)

    gptscan.create_gui()

    clear_filter_cmd = [cmd for text, cmd in button_cmds if text == "×"][-1]
    clear_filter_cmd()

    mock_filter_var.set.assert_called_with("")
    mock_apply_filter.assert_called_once()
    mock_filter_entry.focus_set.assert_called_once()


def test_on_provider_change_model_fallback(monkeypatch):
    mock_provider_var = MagicMock()
    mock_provider_var.get.return_value = "openai"

    mock_api_base_var = MagicMock()
    mock_api_base_var.get.return_value = "http://localhost:11434/v1"

    mock_model_var = MagicMock()
    mock_model_combo = MagicMock()
    mock_model_combo.__getitem__.side_effect = lambda key: [] if key == "values" else MagicMock()

    monkeypatch.setattr(gptscan, "provider_var", mock_provider_var)
    monkeypatch.setattr(gptscan, "api_base_var", mock_api_base_var)
    monkeypatch.setattr(gptscan, "model_var", mock_model_var)
    monkeypatch.setattr(gptscan, "model_combo", mock_model_combo)
    monkeypatch.setattr(gptscan, "update_model_presets", lambda p: None)
    monkeypatch.setattr(gptscan, "toggle_ai_controls", lambda: None)

    gptscan.on_provider_change()

    mock_api_base_var.set.assert_called_once_with("")
    mock_model_var.set.assert_called_with("gpt-4o")
    assert gptscan.Config.provider == "openai"
