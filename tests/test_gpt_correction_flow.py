import pytest
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
import gptscan

# Target: Verify the "correction flow" in async_handle_gpt_response.
# When the initial GPT response is invalid (e.g., missing keys), the code
# should append the error to the message history and ask GPT to correct it.

class MockCompletionResponse:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

class MockAsyncCompletions:
    def __init__(self, responses):
        # responses is a list of content strings or exception objects
        self.responses = iter(responses)
        self.call_args_list = []

    async def create(self, model, messages):
        self.call_args_list.append({"model": model, "messages": list(messages)})
        resp = next(self.responses)
        if isinstance(resp, Exception):
            raise resp
        return MockCompletionResponse(resp)

@pytest.fixture
def mock_openai_client(monkeypatch):
    """Sets up the AsyncOpenAI client mock."""
    client_mock = MagicMock()
    # Ensure get_async_openai_client returns our mock
    monkeypatch.setattr(gptscan, "get_async_openai_client", lambda: client_mock)
    # Clear cache to ensure logic runs
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {})
    return client_mock

@pytest.mark.asyncio
async def test_gpt_correction_flow_success(mock_openai_client):
    """
    Test scenario:
    1. First response is invalid JSON (e.g., missing keys).
    2. Code detects this, sends a correction request.
    3. Second response is valid JSON.
    4. Function returns the valid JSON.
    """

    # invalid: missing 'threat-level'
    resp1_content = '{"administrator": "admin", "end-user": "user"}'
    # valid
    resp2_content = '{"administrator": "admin", "end-user": "user", "threat-level": 10}'

    mock_completions = MockAsyncCompletions([resp1_content, resp2_content])
    mock_openai_client.chat.completions = mock_completions

    limiter = gptscan.AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(1)

    result = await gptscan.async_handle_gpt_response(
        snippet="code_snippet",
        taskdesc="task_description",
        rate_limiter=limiter,
        semaphore=semaphore
    )

    # 1. Verify result is the valid JSON from the second attempt
    assert result is not None
    assert result["threat-level"] == 10

    # 2. Verify interaction flow
    assert len(mock_completions.call_args_list) == 2

    # First call: System prompt + User snippet
    call1 = mock_completions.call_args_list[0]
    assert len(call1["messages"]) == 2
    assert call1["messages"][0]["role"] == "system"
    assert call1["messages"][1]["role"] == "user"
    assert call1["messages"][1]["content"] == "code_snippet"

    # Second call: History + Assistant's bad response + Correction request
    call2 = mock_completions.call_args_list[1]
    assert len(call2["messages"]) == 4
    # Previous messages preserved
    assert call2["messages"][0] == call1["messages"][0]
    assert call2["messages"][1] == call1["messages"][1]
    # New messages
    assert call2["messages"][2]["role"] == "assistant"
    assert call2["messages"][2]["content"] == resp1_content
    assert call2["messages"][3]["role"] == "user"
    assert "I encountered an issue" in call2["messages"][3]["content"]
    assert "Missing keys: threat-level" in call2["messages"][3]["content"]

@pytest.mark.asyncio
async def test_gpt_correction_flow_failure_max_retries(mock_openai_client, monkeypatch):
    """
    Test scenario:
    GPT keeps returning invalid JSON until MAX_RETRIES is exhausted.
    """
    monkeypatch.setattr(gptscan.Config, "MAX_RETRIES", 2)

    # Always invalid
    resp_content = '{"administrator": "admin"}' # Missing keys

    # Provide enough responses to hit the limit
    mock_completions = MockAsyncCompletions([resp_content, resp_content, resp_content])
    mock_openai_client.chat.completions = mock_completions

    limiter = gptscan.AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(1)

    result = await gptscan.async_handle_gpt_response(
        snippet="code_snippet",
        taskdesc="task_description",
        rate_limiter=limiter,
        semaphore=semaphore
    )

    # Should return None after exhausting retries
    assert result is None
    # Called initial + retries. MAX_RETRIES is a bit ambiguous in loop:
    # while retries < MAX_RETRIES.
    # 0 < 2 -> Call 1. Fail. retries=1.
    # 1 < 2 -> Call 2. Fail. retries=2.
    # 2 < 2 -> False. Loop ends.
    # So 2 calls total.
    assert len(mock_completions.call_args_list) == 2
