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
    responses = [_MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 50}')]
    mock_completions = _MockCompletions(responses)

    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=mock_completions)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=MockOpenAI))

    result_first = gptscan.handle_gpt_response("snippet", "task")
    result_second = gptscan.handle_gpt_response("snippet", "task")

    assert result_first == result_second
    assert mock_completions.calls == 1


def test_handle_gpt_response_retries_after_invalid_json(monkeypatch):
    gptscan.Config.gpt_cache = {}
    gptscan.Config.apikey = "dummy"
    invalid_response = _MockResponse('{"administrator": "Admin"}')
    valid_response = _MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 70}')
    mock_completions = _MockCompletions([invalid_response, valid_response])

    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=mock_completions)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=MockOpenAI))
    monkeypatch.setattr(gptscan.Config, "MAX_RETRIES", 3)

    result = gptscan.handle_gpt_response("another snippet", "task")

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

    # Mock handle_gpt_response to track its calls
    gpt_response_called = False
    def mock_handle_gpt_response(*args, **kwargs):
        nonlocal gpt_response_called
        gpt_response_called = True
        return {}
    monkeypatch.setattr(gptscan, "handle_gpt_response", mock_handle_gpt_response)

    # Temporarily disable GPT functionality
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", False)

    # Create a dummy file for scanning
    (tmp_path / "test.txt").write_text("malicious content")
    gptscan.Config.set_extensions([".txt"], missing=False)

    # Execute the scan and consume the generator
    scan_results = list(gptscan.scan_files(scan_path=str(tmp_path), deep_scan=False, show_all=True, use_gpt=True))

    # Verify that handle_gpt_response was not called
    assert not gpt_response_called, "handle_gpt_response should not be called when GPT is disabled"

    # Verify that a result was still produced but without GPT data
    result_data = next((item[1] for item in scan_results if item[0] == 'result'), None)
    assert result_data is not None
    assert result_data[2] == '', "Administrator description should be empty when GPT is disabled"
