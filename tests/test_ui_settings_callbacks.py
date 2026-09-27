from unittest.mock import MagicMock
import pytest
import gptscan


@pytest.fixture(autouse=True)
def restore_config_and_gui_state():
    orig_max_file_size = gptscan.Config.MAX_FILE_SIZE
    orig_threshold = gptscan.Config.THRESHOLD
    orig_root = getattr(gptscan, "root", None)
    orig_tree = getattr(gptscan, "tree", None)
    orig_textbox = getattr(gptscan, "textbox", None)
    orig_filter_var = getattr(gptscan, "filter_var", None)
    orig_filter_entry = getattr(gptscan, "filter_entry", None)
    orig_api_key_entry = getattr(gptscan, "api_key_entry", None)
    orig_api_entry = getattr(gptscan, "api_entry", None)

    yield

    gptscan.Config.MAX_FILE_SIZE = orig_max_file_size
    gptscan.Config.THRESHOLD = orig_threshold
    gptscan.root = orig_root
    gptscan.tree = orig_tree
    gptscan.textbox = orig_textbox
    gptscan.filter_var = orig_filter_var
    gptscan.filter_entry = orig_filter_entry
    gptscan.api_key_entry = orig_api_key_entry
    gptscan.api_entry = orig_api_entry


def create_mock_combo(*args, **kwargs):
    combo = MagicMock()
    combo.cget.return_value = "*"
    combo.get.return_value = "10"
    combo.select_range = MagicMock()
    combo.focus_set = MagicMock()
    combo.delete = MagicMock()
    combo.insert = MagicMock()
    combo.grid = MagicMock()
    return combo


def create_mock_tree(*args, **kwargs):
    tree = MagicMock()
    tree.tag_configure = MagicMock()
    tree.column = MagicMock()
    tree.__getitem__.return_value = (
        "path",
        "own_conf",
        "admin_desc",
        "end-user_desc",
        "gpt_conf",
        "snippet",
        "line",
        "orig_json",
    )
    return tree


def test_on_max_size_change_valid(monkeypatch):
    mock_spinbox = MagicMock()
    mock_spinbox.get.return_value = "15"

    spinboxes = []

    def mock_spinbox_factory(*args, **kwargs):
        sb = mock_spinbox if len(spinboxes) == 0 else MagicMock()
        spinboxes.append((kwargs.get("command"), sb))
        return sb

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "Combobox", create_mock_combo)
    monkeypatch.setattr(gptscan.ttk, "Treeview", create_mock_tree)
    monkeypatch.setattr(gptscan.ttk, "Spinbox", mock_spinbox_factory)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)

    gptscan.create_gui()

    on_max_size_change_cmd = spinboxes[0][0]
    assert callable(on_max_size_change_cmd)

    on_max_size_change_cmd()

    assert gptscan.Config.MAX_FILE_SIZE == 15 * 1024 * 1024


def test_on_max_size_change_invalid_value_error(monkeypatch):
    mock_spinbox = MagicMock()
    mock_spinbox.get.return_value = "invalid_number"

    spinboxes = []

    def mock_spinbox_factory(*args, **kwargs):
        sb = mock_spinbox if len(spinboxes) == 0 else MagicMock()
        spinboxes.append((kwargs.get("command"), sb))
        return sb

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    gptscan.Config.MAX_FILE_SIZE = 10 * 1024 * 1024

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "Combobox", create_mock_combo)
    monkeypatch.setattr(gptscan.ttk, "Treeview", create_mock_tree)
    monkeypatch.setattr(gptscan.ttk, "Spinbox", mock_spinbox_factory)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)

    gptscan.create_gui()

    on_max_size_change_cmd = spinboxes[0][0]
    on_max_size_change_cmd()

    assert gptscan.Config.MAX_FILE_SIZE == 10 * 1024 * 1024


def test_toggle_api_key_visibility_masked_to_unmasked(monkeypatch):
    buttons = {}

    def mock_entry_factory(*args, **kwargs):
        config_dict = {"show": kwargs.get("show", "")}
        e = MagicMock()
        e.__getitem__.side_effect = lambda item: config_dict.get(item, "")
        e.cget.side_effect = lambda item: config_dict.get(item, "")

        def mock_config(**kw):
            config_dict.update(kw)

        e.config.side_effect = mock_config
        e.config_dict = config_dict
        return e

    def mock_button_factory(*args, **kwargs):
        btn = MagicMock()
        text = kwargs.get("text")
        if text in ("Show", "Hide"):
            buttons["show_key_btn"] = (btn, kwargs.get("command"))
        return btn

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "Combobox", create_mock_combo)
    monkeypatch.setattr(gptscan.ttk, "Treeview", create_mock_tree)
    monkeypatch.setattr(gptscan.ttk, "Entry", mock_entry_factory)
    monkeypatch.setattr(gptscan.ttk, "Button", mock_button_factory)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)

    gptscan.create_gui()

    assert "show_key_btn" in buttons
    show_btn_mock, toggle_fn = buttons["show_key_btn"]

    api_key_entry = gptscan.api_key_entry

    assert api_key_entry["show"] == "*"

    toggle_fn()

    assert api_key_entry["show"] == ""
    show_btn_mock.config.assert_called_with(text="Hide")

    toggle_fn()

    assert api_key_entry["show"] == "*"
    show_btn_mock.config.assert_called_with(text="Show")


def test_on_threshold_change_valid_and_clamped(monkeypatch):
    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, "_apply_filter", mock_apply_filter)

    spinboxes = []

    class MockSpinbox(MagicMock):
        def __init__(self, val="50", **kwargs):
            super().__init__(**kwargs)
            self._val = val

        def get(self):
            return self._val

        def set_val(self, val):
            self._val = val

    def mock_spinbox_factory(*args, **kwargs):
        sb = MockSpinbox(**kwargs)
        spinboxes.append((kwargs.get("command"), sb))
        return sb

    mock_root = MagicMock()
    mock_root.after = MagicMock()

    monkeypatch.setattr(gptscan.tk, "Tk", lambda: mock_root)
    monkeypatch.setattr(gptscan.ttk, "Combobox", create_mock_combo)
    monkeypatch.setattr(gptscan.ttk, "Treeview", create_mock_tree)
    monkeypatch.setattr(gptscan.ttk, "Spinbox", mock_spinbox_factory)
    monkeypatch.setattr(gptscan.ttk, "LabelFrame", gptscan.ttk.Frame)

    gptscan.create_gui()

    threshold_cmd, threshold_sb = spinboxes[1]
    assert callable(threshold_cmd)

    # Valid change
    threshold_sb.set_val("70")
    threshold_cmd()
    assert gptscan.Config.THRESHOLD == 70
    assert mock_apply_filter.call_count == 1

    # Clamped underflow
    threshold_sb.set_val("-15")
    threshold_cmd()
    assert gptscan.Config.THRESHOLD == 0

    # Clamped overflow
    threshold_sb.set_val("150")
    threshold_cmd()
    assert gptscan.Config.THRESHOLD == 100

    # Invalid non-numeric
    threshold_sb.set_val("invalid")
    threshold_cmd()
    assert gptscan.Config.THRESHOLD == 100
