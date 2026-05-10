from unittest.mock import MagicMock
import pytest
import gptscan

@pytest.fixture
def mock_gui_ollama(monkeypatch):
    """Setup mocks for AI-related GUI widgets and state."""
    mock_gpt_var = MagicMock()
    mock_provider_combo = MagicMock()
    mock_model_combo = MagicMock()
    mock_api_key_entry = MagicMock()
    mock_api_entry = MagicMock()
    mock_show_key_btn = MagicMock()
    mock_update_tree_columns = MagicMock()
    mock_api_base_var = MagicMock()

    monkeypatch.setattr(gptscan, "gpt_var", mock_gpt_var)
    monkeypatch.setattr(gptscan, "provider_combo", mock_provider_combo)
    monkeypatch.setattr(gptscan, "model_combo", mock_model_combo)
    monkeypatch.setattr(gptscan, "api_key_entry", mock_api_key_entry)
    monkeypatch.setattr(gptscan, "api_entry", mock_api_entry)
    monkeypatch.setattr(gptscan, "show_key_btn", mock_show_key_btn)
    monkeypatch.setattr(gptscan, "update_tree_columns", mock_update_tree_columns)
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    # For on_provider_change
    mock_provider_var = MagicMock()
    monkeypatch.setattr(gptscan, "provider_var", mock_provider_var)
    monkeypatch.setattr(gptscan, "api_base_var", mock_api_base_var)
    mock_model_var = MagicMock()
    monkeypatch.setattr(gptscan, "model_var", mock_model_var)

    # Mock update_model_presets
    monkeypatch.setattr(gptscan, "update_model_presets", MagicMock())

    return {
        "gpt_var": mock_gpt_var,
        "provider_combo": mock_provider_combo,
        "model_combo": mock_model_combo,
        "api_key_entry": mock_api_key_entry,
        "api_entry": mock_api_entry,
        "show_key_btn": mock_show_key_btn,
        "update_tree_columns": mock_update_tree_columns,
        "provider_var": mock_provider_var,
        "api_base_var": mock_api_base_var,
        "model_var": mock_model_var
    }

def test_toggle_ai_controls_ollama_disabled_key(mock_gui_ollama, monkeypatch):
    """When provider is ollama, api_key_entry and show_key_btn should be disabled even if AI is enabled."""
    mock_gui_ollama["gpt_var"].get.return_value = True
    mock_gui_ollama["provider_combo"].get.return_value = "ollama"
    monkeypatch.setattr(gptscan, "current_cancel_event", None)

    gptscan.toggle_ai_controls()

    mock_gui_ollama["api_key_entry"].config.assert_called_with(state="disabled")
    mock_gui_ollama["show_key_btn"].config.assert_called_with(state="disabled")
    mock_gui_ollama["api_entry"].config.assert_called_with(state="normal")

def test_on_provider_change_to_ollama_sets_default_base(mock_gui_ollama):
    """Switching to ollama should set default API base if empty."""
    mock_gui_ollama["provider_var"].get.return_value = "ollama"
    mock_gui_ollama["api_base_var"].get.return_value = ""

    gptscan.on_provider_change(None)

    mock_gui_ollama["api_base_var"].set.assert_called_with("http://localhost:11434/v1")

def test_on_provider_change_from_ollama_clears_default_base(mock_gui_ollama):
    """Switching away from ollama should clear default API base if it matches the default."""
    mock_gui_ollama["provider_var"].get.return_value = "openai"
    mock_gui_ollama["api_base_var"].get.return_value = "http://localhost:11434/v1"

    gptscan.on_provider_change(None)

    mock_gui_ollama["api_base_var"].set.assert_called_with("")
