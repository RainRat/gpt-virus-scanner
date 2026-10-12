import pytest
import gptscan


class MockWidget:
    def __init__(self, value=""):
        self.value = value
        self.focused = False
        self.deleted = False

    def get(self):
        return self.value

    def delete(self, start, end):
        self.value = ""
        self.deleted = True

    def focus_set(self):
        self.focused = True


def test_on_target_escape_active_cancel(monkeypatch):
    monkeypatch.setattr(gptscan, "current_cancel_event", object())
    cancelled = False

    def mock_cancel():
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr(gptscan, "cancel_scan", mock_cancel)

    result = gptscan.on_target_escape()
    assert result == "break"
    assert cancelled is True


def test_on_target_escape_with_text(monkeypatch):
    monkeypatch.setattr(gptscan, "current_cancel_event", None)
    mock_box = MockWidget("   some path   ")
    monkeypatch.setattr(gptscan, "textbox", mock_box)

    visibility_updated = False

    def mock_update_visibility():
        nonlocal visibility_updated
        visibility_updated = True

    monkeypatch.setattr(gptscan, "update_clear_target_visibility", mock_update_visibility)

    result = gptscan.on_target_escape()
    assert result == "break"
    assert mock_box.get() == ""
    assert mock_box.focused is True
    assert visibility_updated is True


def test_on_target_escape_with_whitespace_only(monkeypatch):
    monkeypatch.setattr(gptscan, "current_cancel_event", None)
    mock_box = MockWidget("   ")
    monkeypatch.setattr(gptscan, "textbox", mock_box)

    visibility_updated = False

    def mock_update_visibility():
        nonlocal visibility_updated
        visibility_updated = True

    monkeypatch.setattr(gptscan, "update_clear_target_visibility", mock_update_visibility)

    result = gptscan.on_target_escape()
    assert result == "break"
    assert mock_box.get() == ""
    assert mock_box.focused is True
    assert visibility_updated is True


def test_on_target_escape_empty_passes_to_root_escape(monkeypatch):
    monkeypatch.setattr(gptscan, "current_cancel_event", None)
    mock_box = MockWidget("")
    monkeypatch.setattr(gptscan, "textbox", mock_box)

    root_escape_called = False

    def mock_on_root_escape(event=None):
        nonlocal root_escape_called
        root_escape_called = True
        return ""

    monkeypatch.setattr(gptscan, "on_root_escape", mock_on_root_escape)

    result = gptscan.on_target_escape()
    assert result == ""
    assert root_escape_called is True
