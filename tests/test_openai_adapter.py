import pytest
import sys
from unittest.mock import MagicMock
from types import ModuleType
import gptscan

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
    mock_async_cls.assert_called_once_with(api_key="dummy_key", base_url=None)

def test_get_async_openai_client_returns_none_if_no_api_key(monkeypatch):
    monkeypatch.setattr(gptscan, "_async_openai_client", None)
    monkeypatch.setattr(gptscan.Config, "apikey", "")

    assert gptscan.get_async_openai_client() is None
