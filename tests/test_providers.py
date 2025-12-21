import pytest
import sys
from unittest.mock import MagicMock
from types import ModuleType
import gptscan

def test_ollama_client_config(monkeypatch):
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "")
    monkeypatch.setattr(gptscan.Config, "provider", "ollama")
    monkeypatch.setattr(gptscan.Config, "api_base", None)

    mock_async_cls = MagicMock()
    mock_openai_mod = ModuleType("openai")
    mock_openai_mod.AsyncOpenAI = mock_async_cls
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

    client = gptscan.get_async_openai_client()

    assert client is mock_async_cls.return_value
    mock_async_cls.assert_called_once_with(api_key="ollama", base_url="http://localhost:11434/v1")

def test_openrouter_client_config(monkeypatch):
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "my_key")
    monkeypatch.setattr(gptscan.Config, "provider", "openrouter")
    monkeypatch.setattr(gptscan.Config, "api_base", None)

    mock_async_cls = MagicMock()
    mock_openai_mod = ModuleType("openai")
    mock_openai_mod.AsyncOpenAI = mock_async_cls
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

    client = gptscan.get_async_openai_client()

    assert client is mock_async_cls.return_value
    mock_async_cls.assert_called_once_with(api_key="my_key", base_url="https://openrouter.ai/api/v1")

def test_custom_api_base(monkeypatch):
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "my_key")
    monkeypatch.setattr(gptscan.Config, "provider", "openai")
    monkeypatch.setattr(gptscan.Config, "api_base", "https://custom.api/v1")

    mock_async_cls = MagicMock()
    mock_openai_mod = ModuleType("openai")
    mock_openai_mod.AsyncOpenAI = mock_async_cls
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

    client = gptscan.get_async_openai_client()

    assert client is mock_async_cls.return_value
    mock_async_cls.assert_called_once_with(api_key="my_key", base_url="https://custom.api/v1")
