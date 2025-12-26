import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from types import SimpleNamespace
import gptscan
from gptscan import extract_data_from_gpt_response, AsyncRateLimiter

@pytest.fixture
def mock_response():
    def _create(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
    return _create

def test_extract_data_non_dict(mock_response):
    response = mock_response('[1, 2, 3]')
    result = extract_data_from_gpt_response(response)
    assert result == "Response JSON must be an object with expected keys."

def test_extract_data_extra_keys(mock_response):
    # keys are: administrator, end-user, threat-level
    content = '{"administrator": "a", "end-user": "b", "threat-level": 10, "extra": "x"}'
    response = mock_response(content)
    result = extract_data_from_gpt_response(response)
    assert "Unexpected keys present: extra" in result

def test_extract_data_threat_level_type_error(mock_response):
    content = '{"administrator": "a", "end-user": "b", "threat-level": "high"}'
    response = mock_response(content)
    result = extract_data_from_gpt_response(response)
    assert "not a valid integer" in result

def test_extract_data_threat_level_bounds(mock_response):
    content = '{"administrator": "a", "end-user": "b", "threat-level": 150}'
    response = mock_response(content)
    result = extract_data_from_gpt_response(response)
    assert "not between 0 and 100" in result

class MockRateLimitError(Exception):
    def __init__(self):
        self.status_code = 429
        self.status = "429 Rate Limit"

class MockCompletionsFailure:
    def __init__(self, failures, success_content):
        self.failures = failures # List of exceptions to raise
        self.success_content = success_content
        self.call_count = 0

    async def create(self, model, messages):
        self.call_count += 1
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.success_content))]
        )

@pytest.mark.asyncio
async def test_async_handle_gpt_response_retries_on_rate_limit(monkeypatch):
    # Setup - use monkeypatch for globals to avoid pollution
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {})
    monkeypatch.setattr(gptscan.Config, "apikey", "test_key")
    monkeypatch.setattr(gptscan, "_async_openai_client", None)

    valid_json = '{"administrator": "A", "end-user": "U", "threat-level": 10}'

    # Fail twice with rate limit, then succeed
    mock_handler = MockCompletionsFailure(
        [MockRateLimitError(), MockRateLimitError()],
        valid_json
    )

    class MockAsyncOpenAI:
        def __init__(self, api_key, **kwargs):
            self.chat = SimpleNamespace(completions=mock_handler)

    monkeypatch.setattr(gptscan, "get_async_openai_client", lambda: MockAsyncOpenAI("k"))

    # Mock sleep to run fast
    monkeypatch.setattr(asyncio, "sleep",  AsyncMock())

    limiter = MagicMock()
    limiter.acquire = AsyncMock()
    semaphore = asyncio.Semaphore(5)

    result = await gptscan.async_handle_gpt_response(
        "snippet", "task", limiter, semaphore
    )

    assert result["threat-level"] == 10
    assert mock_handler.call_count == 3
