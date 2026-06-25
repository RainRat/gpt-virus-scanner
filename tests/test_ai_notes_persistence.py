import pytest
import threading
import asyncio
from unittest.mock import patch, MagicMock
from gptscan import batch_ai_analysis_events, Config

@pytest.mark.asyncio
async def test_batch_ai_analysis_preserves_existing_notes():
    # Setup mock request with existing notes
    request = {
        "path": "test.py",
        "percent": "50%",
        "snippet": "print('hello')",
        "cleaned_snippet": "print('hello')",
        "line": 1,
        "item_id": "item1",
        "admin_desc": "[Filename Warning] Deceptive name",
        "user_desc": "Caution: Deceptive name"
    }

    mock_json_data = {
        "administrator": "AI analysis of code.",
        "end-user": "Safe to run.",
        "threat-level": 10
    }

    cancel_event = threading.Event()

    # Mock Config and async_handle_gpt_response
    with patch("gptscan.Config.GPT_ENABLED", True), \
         patch("gptscan.Config.taskdesc", "test task"), \
         patch("gptscan.async_handle_gpt_response", return_value=mock_json_data):

        # We need to run the generator and collect events
        events = []
        # batch_ai_analysis_events uses a thread and a queue internally.
        # It's a generator that yields events from the queue.
        gen = batch_ai_analysis_events([request], cancel_event)

        for event_type, data in gen:
            if event_type == 'result':
                events.append(data)
            if event_type == 'summary':
                break

    assert len(events) == 1
    result_data = events[0]
    # result_data format: (path, own_conf, admin, user, gpt, snippet, line, item_id)

    admin_result = result_data[2]
    user_result = result_data[3]

    # If it overwrites, it will not contain the warning
    assert "[Filename Warning]" in admin_result
    assert "Caution:" in user_result
    assert "AI analysis" in admin_result
    assert "Safe to run" in user_result
