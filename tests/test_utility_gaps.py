import asyncio
import pytest
from unittest.mock import MagicMock, patch
from gptscan import format_bytes, extract_data_from_gpt_response, async_handle_gpt_response, Config, AsyncRateLimiter

# --- format_bytes Tests ---

def test_format_bytes_units():
    assert "100.0 B" in format_bytes(100)
    assert "1.0 KiB" in format_bytes(1024)
    assert "1.0 MiB" in format_bytes(1024**2)
    assert "1.0 GiB" in format_bytes(1024**3)
    assert "1.0 TiB" in format_bytes(1024**4)
    assert "1.0 PiB" in format_bytes(1024**5)
    assert "1.5 PiB" in format_bytes(1024**5 * 1.5)

def test_format_bytes_edge_cases():
    assert "0.0 B" in format_bytes(0)
    assert "-512.0 B" in format_bytes(-512)
    assert "-1.0 KiB" in format_bytes(-1024)

# --- extract_data_from_gpt_response Tests ---

def test_extract_data_from_gpt_response_malformed_json():
    # Mock response object
    class MockChoice:
        def __init__(self, content):
            self.message = MagicMock(content=content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    # Malformed JSON (missing closing brace)
    response = MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 50')

    result = extract_data_from_gpt_response(response)

    # Should return the error message from JSONDecodeError
    assert isinstance(result, str)
    assert "Expecting" in result or "column" in result

# --- async_handle_gpt_response Tests ---

@pytest.mark.asyncio
async def test_async_handle_gpt_response_no_client(monkeypatch):
    # Mock get_async_openai_client to return None
    monkeypatch.setattr("gptscan.get_async_openai_client", lambda: None)

    # Ensure cache is empty and doesn't hit
    monkeypatch.setattr(Config, "gpt_cache", {})

    limiter = AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(5)

    result = await async_handle_gpt_response(
        "some snippet",
        "some task",
        rate_limiter=limiter,
        semaphore=semaphore
    )

    assert result is None
