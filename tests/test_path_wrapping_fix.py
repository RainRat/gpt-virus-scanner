import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import threading

def test_rescan_path_with_spaces_and_wrapping():
    # Mock tree
    gptscan.tree = MagicMock()
    gptscan.tree.selection.return_value = ("item1",)
    gptscan.tree.exists.return_value = True
    gptscan.current_cancel_event = None

    # Path with space that was wrapped
    original_path = "C:/My Documents/script.py"
    wrapped_path = "C:/My\nDocuments/script.py"

    # 1. Test with orig_json (the preferred way)
    orig_values = [original_path, "80%", "Admin", "User", "90%", "Snippet"]
    vals_with_json = [wrapped_path, "80%", "Admin", "User", "90%", "Snippet", json.dumps(orig_values)]

    def tree_item_side_effect(item_id, option=None):
        if option == "values":
            return vals_with_json
        return {"values": vals_with_json}

    gptscan.tree.item.side_effect = tree_item_side_effect

    with patch("threading.Thread") as mock_thread, \
         patch("gptscan.set_scanning_state"), \
         patch("gptscan.update_status"):
        gptscan.rescan_selected()

        _, kwargs = mock_thread.call_args
        thread_args = kwargs["args"]
        # Should have restored exactly the original path
        assert thread_args[0] == [original_path]
        assert thread_args[1] == {original_path: "item1"}

    # Reset cancel event for second test
    gptscan.current_cancel_event = None

    # 2. Test fallback (when orig_json is missing)
    vals_no_json = [wrapped_path, "80%", "Admin", "User", "90%", "Snippet", ""]

    def tree_item_side_effect_no_json(item_id, option=None):
        if option == "values":
            return vals_no_json
        return {"values": vals_no_json}

    gptscan.tree.item.side_effect = tree_item_side_effect_no_json

    with patch("threading.Thread") as mock_thread, \
         patch("gptscan.set_scanning_state"), \
         patch("gptscan.update_status"):
        gptscan.rescan_selected()

        assert mock_thread.called
        _, kwargs = mock_thread.call_args
        thread_args = kwargs["args"]
        # Fallback should restore space instead of just removing newline
        assert thread_args[0] == [original_path]
