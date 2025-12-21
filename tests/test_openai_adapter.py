import pytest
import asyncio
import sys
from unittest.mock import MagicMock
from types import ModuleType
import gptscan

# Test the adapter classes directly
@pytest.mark.asyncio
async def test_adapter_calls_sync_client_via_thread():
    # Mock the sync client structure
    mock_sync_client = MagicMock()
    mock_response = {"choices": [{"message": {"content": "test"}}]}

    mock_sync_client.chat.completions.create.return_value = mock_response

    adapter = gptscan._SyncToAsyncOpenAIAdapter(mock_sync_client)

    assert isinstance(adapter.chat, gptscan._SyncChatAdapter)
    assert isinstance(adapter.chat.completions, gptscan._SyncCompletionsAdapter)

    kwargs = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]}
    result = await adapter.chat.completions.create(**kwargs)

    assert result == mock_response
    mock_sync_client.chat.completions.create.assert_called_once_with(**kwargs)


def test_get_async_openai_client_uses_native_async(monkeypatch):
    """Verify that if AsyncOpenAI is available, it is used directly."""
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "dummy_key")

    mock_async_cls = MagicMock()
    mock_openai_mod = ModuleType("openai")
    mock_openai_mod.AsyncOpenAI = mock_async_cls

    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

    client = gptscan.get_async_openai_client()

    assert client is mock_async_cls.return_value
    mock_async_cls.assert_called_once_with(api_key="dummy_key")


def test_get_async_openai_client_fallback_to_adapter(monkeypatch):
    """Verify fallback to adapter when AsyncOpenAI is missing."""
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan, "_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "dummy_key")

    # Create a mock module WITHOUT AsyncOpenAI, but WITH OpenAI (sync)
    mock_sync_cls = MagicMock()
    mock_openai_mod = ModuleType("openai")
    mock_openai_mod.OpenAI = mock_sync_cls

    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

    client = gptscan.get_async_openai_client()

    assert isinstance(client, gptscan._SyncToAsyncOpenAIAdapter)

    mock_sync_cls.assert_called_with(api_key="dummy_key")
    assert client.chat.completions._client == mock_sync_cls.return_value

def test_get_async_openai_client_returns_none_if_no_api_key(monkeypatch):
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "")

    assert gptscan.get_async_openai_client() is None
