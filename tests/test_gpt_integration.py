import asyncio
import sys
from types import SimpleNamespace

import gptscan


class _MockCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def create(self, model, messages):
        self.calls += 1
        return next(self.responses)


class _MockResponse:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def test_handle_gpt_response_returns_cached_result(monkeypatch):
    gptscan.Config.gpt_cache = {}
    gptscan.Config.apikey = "dummy"
    gptscan._openai_client = None
    gptscan._async_openai_client = None
    responses = [_MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 50}')]
    mock_completions = _MockCompletions(responses)

    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=mock_completions)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=MockOpenAI))

    limiter = gptscan.AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(5)

    result_first = asyncio.run(
        gptscan.async_handle_gpt_response("snippet", "task", limiter, semaphore)
    )
    result_second = asyncio.run(
        gptscan.async_handle_gpt_response("snippet", "task", limiter, semaphore)
    )

    assert result_first == result_second
    assert mock_completions.calls == 1


def test_handle_gpt_response_retries_after_invalid_json(monkeypatch):
    gptscan.Config.gpt_cache = {}
    gptscan.Config.apikey = "dummy"
    gptscan._openai_client = None
    gptscan._async_openai_client = None
    invalid_response = _MockResponse('{"administrator": "Admin"}')
    valid_response = _MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 70}')
    mock_completions = _MockCompletions([invalid_response, valid_response])

    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=mock_completions)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=MockOpenAI))
    monkeypatch.setattr(gptscan.Config, "MAX_RETRIES", 3)

    limiter = gptscan.AsyncRateLimiter(60)
    semaphore = asyncio.Semaphore(5)

    result = asyncio.run(
        gptscan.async_handle_gpt_response(
            "another snippet", "task", limiter, semaphore
        )
    )

    assert result["threat-level"] == 70
    assert mock_completions.calls == 2


def test_scan_files_does_not_call_gpt_when_disabled(monkeypatch, tmp_path):
    # Mock TensorFlow and the model to simulate a high-confidence scan result
    mock_model = SimpleNamespace(predict=lambda *args, **kwargs: [[0.9]])
    mock_keras = SimpleNamespace(models=SimpleNamespace(load_model=lambda *args, **kwargs: mock_model))

    class MockTensor:
        def __init__(self, data):
            self.data = [data]

        def __getitem__(self, key):
            # Handle the multi-dimensional slicing used in the code
            row_slice, col_slice = key
            return self.data[0][col_slice]

    mock_tf = SimpleNamespace(
        keras=mock_keras,
        constant=lambda data: data,
        expand_dims=lambda data, axis: MockTensor(data)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

    # Mock async_handle_gpt_response to track its calls
    gpt_response_called = False

    async def mock_async_handle_gpt_response(*args, **kwargs):
        nonlocal gpt_response_called
        gpt_response_called = True
        return {}

    monkeypatch.setattr(gptscan, "async_handle_gpt_response", mock_async_handle_gpt_response)

    # Temporarily disable GPT functionality
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", False)

    # Create a dummy file for scanning
    (tmp_path / "test.txt").write_text("malicious content")
    gptscan.Config.set_extensions([".txt"], missing=False)

    # Execute the scan and consume the generator
    scan_results = list(gptscan.scan_files(scan_path=str(tmp_path), deep_scan=False, show_all=True, use_gpt=True))

    # Verify that async_handle_gpt_response was not called
    assert not gpt_response_called, "async_handle_gpt_response should not be called when GPT is disabled"

    # Verify that a result was still produced but without GPT data
    result_data = next((item[1] for item in scan_results if item[0] == 'result'), None)
    assert result_data is not None
    assert result_data[2] == '', "Administrator description should be empty when GPT is disabled"
