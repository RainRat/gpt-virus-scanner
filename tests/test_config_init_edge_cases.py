import os
import pytest
from unittest.mock import patch
from gptscan import Config

@pytest.fixture
def clean_config():
    """Reset Config state for initialization tests."""
    orig_apikey = Config.apikey
    orig_taskdesc = Config.taskdesc
    orig_gpt_enabled = Config.GPT_ENABLED

    # Set to empty defaults
    Config.apikey = ""
    Config.taskdesc = ""
    Config.GPT_ENABLED = False

    yield Config

    # Restore
    Config.apikey = orig_apikey
    Config.taskdesc = orig_taskdesc
    Config.GPT_ENABLED = orig_gpt_enabled

def test_initialize_picks_up_openai_env_key(clean_config):
    """Verify that OPENAI_API_KEY is used if apikey.txt is empty."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-openai"}):
        with patch("gptscan.load_file", return_value=[]): # Mock extensions etc
            clean_config.initialize()
            assert clean_config.apikey == "sk-test-openai"

def test_initialize_picks_up_openrouter_env_key(clean_config):
    """Verify that OPENROUTER_API_KEY is used if apikey.txt and OPENAI_API_KEY are empty."""
    # Ensure OPENAI_API_KEY is NOT in environ for this test
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-openrouter"}, clear=True):
        with patch("gptscan.load_file", return_value=[]):
            clean_config.initialize()
            assert clean_config.apikey == "sk-test-openrouter"

def test_gpt_enabled_true_if_taskdesc_present(clean_config):
    """Verify GPT_ENABLED is True when taskdesc is loaded."""
    clean_config.taskdesc = "Analyze this code"
    with patch("gptscan.load_file", return_value=[]):
        clean_config.initialize()
        assert clean_config.GPT_ENABLED is True

def test_gpt_enabled_false_if_taskdesc_missing(clean_config):
    """Verify GPT_ENABLED is False when taskdesc is missing."""
    clean_config.taskdesc = ""
    with patch("gptscan.load_file", return_value=[]):
        clean_config.initialize()
        assert clean_config.GPT_ENABLED is False
