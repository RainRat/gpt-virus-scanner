import sys
import unittest.mock as mock
import pytest

# gptscan is imported after conftest mocks tkinter
import gptscan

def test_bind_hover_message():
    # Setup mocks
    mock_widget = mock.Mock()
    gptscan.status_label = mock.Mock()
    gptscan.root = mock.Mock()
    gptscan.current_cancel_event = None

    # Helper to capture bindings
    bindings = {}
    def bind_side_effect(event, handler):
        bindings[event] = handler
    mock_widget.bind.side_effect = bind_side_effect

    # Call the function
    message = "Test Message"
    gptscan.bind_hover_message(mock_widget, message)

    # Verify bindings were set
    assert "<Enter>" in bindings
    assert "<Leave>" in bindings

    # Test Enter event (Idle) - Restore Logic
    gptscan.current_cancel_event = None
    gptscan.status_label.cget.return_value = "Previous Status" # Setup previous status

    bindings["<Enter>"](None)

    gptscan.status_label.cget.assert_called_with("text")
    gptscan.status_label.config.assert_called_with(text=message)
    gptscan.root.update_idletasks.assert_called()

    # Test Leave event (Idle) - Should restore "Previous Status"
    gptscan.status_label.config.reset_mock()
    gptscan.root.update_idletasks.reset_mock()

    bindings["<Leave>"](None)

    gptscan.status_label.config.assert_called_with(text="Previous Status")

    # Test Enter event (Busy)
    gptscan.status_label.config.reset_mock()
    gptscan.current_cancel_event = mock.Mock() # Not None
    bindings["<Enter>"](None)
    gptscan.status_label.config.assert_not_called()
