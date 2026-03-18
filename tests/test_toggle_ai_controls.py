from unittest.mock import MagicMock, patch
import pytest
import gptscan

@pytest.fixture
def mock_gui_ai(monkeypatch):
    """Setup mocks for AI-related GUI widgets and state."""
    mock_gpt_var = MagicMock()
    mock_provider_combo = MagicMock()
    mock_model_combo = MagicMock()
    mock_update_tree_columns = MagicMock()

    monkeypatch.setattr(gptscan, "gpt_var", mock_gpt_var)
    monkeypatch.setattr(gptscan, "provider_combo", mock_provider_combo)
    monkeypatch.setattr(gptscan, "model_combo", mock_model_combo)
    monkeypatch.setattr(gptscan, "update_tree_columns", mock_update_tree_columns)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    return {
        "gpt_var": mock_gpt_var,
        "provider_combo": mock_provider_combo,
        "model_combo": mock_model_combo,
        "update_tree_columns": mock_update_tree_columns
    }

def test_toggle_ai_controls_enabled_no_scan(mock_gui_ai, monkeypatch):
    """AI is on, no scan running: combos should be active."""
    mock_gui_ai["gpt_var"].get.return_value = True
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    gptscan.toggle_ai_controls()

    mock_gui_ai["provider_combo"].config.assert_called_with(state="readonly")
    mock_gui_ai["model_combo"].config.assert_called_with(state="normal")
    mock_gui_ai["update_tree_columns"].assert_called_once()

def test_toggle_ai_controls_disabled_no_scan(mock_gui_ai, monkeypatch):
    """AI is off, no scan running: combos should be disabled."""
    mock_gui_ai["gpt_var"].get.return_value = False
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    gptscan.toggle_ai_controls()

    mock_gui_ai["provider_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["model_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["update_tree_columns"].assert_called_once()

def test_toggle_ai_controls_enabled_during_scan(mock_gui_ai, monkeypatch):
    """AI is on, scan is running: combos should be disabled."""
    mock_gui_ai["gpt_var"].get.return_value = True
    monkeypatch.setattr(gptscan, "current_cancel_event", MagicMock()) # Scan active

    gptscan.toggle_ai_controls()

    mock_gui_ai["provider_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["model_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["update_tree_columns"].assert_called_once()

def test_toggle_ai_controls_disabled_during_scan(mock_gui_ai, monkeypatch):
    """AI is off, scan is running: combos should be disabled."""
    mock_gui_ai["gpt_var"].get.return_value = False
    monkeypatch.setattr(gptscan, "current_cancel_event", MagicMock())

    gptscan.toggle_ai_controls()

    mock_gui_ai["provider_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["model_combo"].config.assert_called_with(state="disabled")
    mock_gui_ai["update_tree_columns"].assert_called_once()

def test_toggle_ai_controls_no_widgets(monkeypatch):
    """Test that function doesn't crash if widgets are None."""
    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = True
    monkeypatch.setattr(gptscan, "gpt_var", mock_gpt_var)
    monkeypatch.setattr(gptscan, "provider_combo", None)
    monkeypatch.setattr(gptscan, "model_combo", None)

    mock_update_tree_columns = MagicMock()
    monkeypatch.setattr(gptscan, "update_tree_columns", mock_update_tree_columns)

    # Should not raise exception
    gptscan.toggle_ai_controls()
    mock_update_tree_columns.assert_called_once()

def test_toggle_ai_controls_no_gpt_var(monkeypatch):
    """Test behavior when gpt_var is None (AI disabled by config)."""
    monkeypatch.setattr(gptscan, "gpt_var", None)
    mock_provider_combo = MagicMock()
    mock_model_combo = MagicMock()
    monkeypatch.setattr(gptscan, "provider_combo", mock_provider_combo)
    monkeypatch.setattr(gptscan, "model_combo", mock_model_combo)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    mock_update_tree_columns = MagicMock()
    monkeypatch.setattr(gptscan, "update_tree_columns", mock_update_tree_columns)

    gptscan.toggle_ai_controls()

    # Should default to disabled since enabled = False
    mock_provider_combo.config.assert_called_with(state="disabled")
    mock_model_combo.config.assert_called_with(state="disabled")
    mock_update_tree_columns.assert_called_once()
