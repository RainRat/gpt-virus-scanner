import sys
from types import SimpleNamespace

import gptscan


class _MockChatCompletion:
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
    gptscan.gpt_cache = {}
    gptscan.apikey = "dummy"
    responses = [_MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 50}')]
    mock_chat = _MockChatCompletion(responses)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(ChatCompletion=mock_chat))

    result_first = gptscan.handle_gpt_response("snippet", "task")
    result_second = gptscan.handle_gpt_response("snippet", "task")

    assert result_first == result_second
    assert mock_chat.calls == 1


def test_handle_gpt_response_retries_after_invalid_json(monkeypatch):
    gptscan.gpt_cache = {}
    gptscan.apikey = "dummy"
    invalid_response = _MockResponse('{"administrator": "Admin"}')
    valid_response = _MockResponse('{"administrator": "Admin", "end-user": "User", "threat-level": 70}')
    mock_chat = _MockChatCompletion([invalid_response, valid_response])
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(ChatCompletion=mock_chat))
    monkeypatch.setattr(gptscan, "MAX_RETRIES", 3)

    result = gptscan.handle_gpt_response("another snippet", "task")

    assert result["threat-level"] == 70
    assert mock_chat.calls == 2
