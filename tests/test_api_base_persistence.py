import os
import json
import pytest
from unittest.mock import MagicMock, patch
from gptscan import Config

@pytest.fixture
def temp_settings_file(tmp_path):
    settings_file = tmp_path / ".gptscan_settings.json"
    with patch('gptscan.Config.SETTINGS_FILE', str(settings_file)):
        yield settings_file

def test_api_base_persistence(temp_settings_file):
    # Initial state
    Config.api_base = None

    # Set a custom API base
    custom_url = "http://localhost:11434/v1"
    Config.api_base = custom_url

    # Save settings
    Config.save_settings()

    # Verify file content
    assert temp_settings_file.exists()
    with open(temp_settings_file, 'r') as f:
        settings = json.load(f)
        assert settings.get("api_base") == custom_url

    # Reset and Load settings
    Config.api_base = "something-else"
    Config.load_settings()

    # Verify loaded value
    assert Config.api_base == custom_url

def test_api_base_empty_persistence(temp_settings_file):
    # Set and Save an empty/None API base
    Config.api_base = None
    Config.save_settings()

    with open(temp_settings_file, 'r') as f:
        settings = json.load(f)
        assert settings.get("api_base") is None

    # Load and verify
    Config.api_base = "should-be-overwritten"
    Config.load_settings()
    assert Config.api_base is None
