import queue
import pytest
from unittest.mock import MagicMock
import gptscan

def test_process_ui_queue_reschedules_on_error(monkeypatch):
    """Verify that process_ui_queue reschedules itself even if a task raises an exception."""
    mock_root = MagicMock()
    mock_queue = queue.Queue()

    # Setup mocks
    monkeypatch.setattr(gptscan, "root", mock_root)
    monkeypatch.setattr(gptscan, "ui_queue", mock_queue)

    def buggy_func():
        raise ValueError("Simulated UI update error")

    # Add a buggy task and a normal task to the queue
    mock_queue.put((buggy_func, (), {}))

    # Running process_ui_queue should propagate the exception from buggy_func
    with pytest.raises(ValueError, match="Simulated UI update error"):
        gptscan.process_ui_queue()

    # Verify that root.after was still called despite the exception
    mock_root.after.assert_called_once_with(50, gptscan.process_ui_queue)
    # Verify that task_done was called (via the finally block in the loop)
    # Actually, the exception happens inside the try block, then finally: ui_queue.task_done() runs.
    # But wait, if it propagates out of process_ui_queue, the while loop terminates.
