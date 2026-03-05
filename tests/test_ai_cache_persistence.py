import os
import json
import pytest
import hashlib
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from gptscan import Config, async_handle_gpt_response, AsyncRateLimiter

@pytest.fixture
def temp_cache_file(tmp_path):
    cache_file = tmp_path / ".gptscan_cache.json"
    original_cache_file = Config.CACHE_FILE
    Config.CACHE_FILE = str(cache_file)
    yield cache_file
    Config.CACHE_FILE = original_cache_file
    if cache_file.exists():
        os.remove(cache_file)

def test_config_save_load_cache(temp_cache_file):
    Config.gpt_cache = {"test_key": {"administrator": "test", "end-user": "test", "threat-level": 10}}
    Config.save_cache()

    assert temp_cache_file.exists()

    # Reset cache and load
    Config.gpt_cache = {}
    Config.load_cache()

    assert "test_key" in Config.gpt_cache
    assert Config.gpt_cache["test_key"]["threat-level"] == 10

@pytest.mark.asyncio
async def test_async_handle_gpt_response_caching(temp_cache_file, monkeypatch):
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "administrator": "Technical analysis",
        "end-user": "Safe",
        "threat-level": 0
    })

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("gptscan.get_async_openai_client", lambda: mock_client)

    # Ensure cache is empty
    Config.gpt_cache = {}
    if temp_cache_file.exists():
        os.remove(temp_cache_file)

    rate_limiter = AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(1)

    snippet = "print('hello')"
    taskdesc = "Analyze this"

    # First call - should trigger API call
    result1 = await async_handle_gpt_response(snippet, taskdesc, rate_limiter, semaphore)
    assert mock_client.chat.completions.create.call_count == 1
    assert result1["threat-level"] == 0
    assert temp_cache_file.exists()

    # Second call - should use cache
    result2 = await async_handle_gpt_response(snippet, taskdesc, rate_limiter, semaphore)
    assert mock_client.chat.completions.create.call_count == 1  # Still 1
    assert result2 == result1

    # Verify cache key includes provider and model
    original_model = Config.model_name
    Config.model_name = "different-model"

    result3 = await async_handle_gpt_response(snippet, taskdesc, rate_limiter, semaphore)
    assert mock_client.chat.completions.create.call_count == 2

    Config.model_name = original_model

def test_clear_ai_cache(temp_cache_file):
    Config.gpt_cache = {"key": "value"}
    Config.save_cache()
    assert temp_cache_file.exists()

    from gptscan import clear_ai_cache
    # Mock update_status to avoid GUI issues
    with patch("gptscan.update_status"):
        clear_ai_cache()

    assert Config.gpt_cache == {}
    Config.load_cache()
    assert Config.gpt_cache == {}
