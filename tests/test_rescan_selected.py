import pytest
from unittest.mock import MagicMock, patch
import gptscan
import threading

@pytest.fixture(autouse=True)
def reset_globals():
    # Store original values
    orig_tree = gptscan.tree
    orig_deep = gptscan.deep_var
    orig_gpt = gptscan.gpt_var
    orig_dry = gptscan.dry_var
    orig_cancel = gptscan.current_cancel_event

    yield

    # Restore original values
    gptscan.tree = orig_tree
    gptscan.deep_var = orig_deep
    gptscan.gpt_var = orig_gpt
    gptscan.dry_var = orig_dry
    gptscan.current_cancel_event = orig_cancel

def test_rescan_selected_logic():
    # Mock tree
    gptscan.tree = MagicMock()
    gptscan.tree.selection.return_value = ("item1",)

    def tree_item_side_effect(item_id, option=None):
        # Using \n to simulate wrapping, but it should be correctly removed
        vals = ("test.py\n", "50%", "", "", "", "snippet")
        if option == "values":
            return vals
        return {"values": vals}

    gptscan.tree.item.side_effect = tree_item_side_effect

    # Mock vars
    gptscan.deep_var = MagicMock()
    gptscan.deep_var.get.return_value = True
    gptscan.gpt_var = MagicMock()
    gptscan.gpt_var.get.return_value = False
    gptscan.dry_var = MagicMock()
    gptscan.dry_var.get.return_value = False

    gptscan.current_cancel_event = None

    with patch("threading.Thread") as mock_thread, \
         patch("gptscan.set_scanning_state"), \
         patch("gptscan.update_status"):
        gptscan.rescan_selected()

        # Verify thread started
        mock_thread.assert_called_once()
        _, kwargs = mock_thread.call_args
        assert kwargs["target"] == gptscan.run_rescan
        thread_args = kwargs["args"]
        assert thread_args[0] == ["test.py"]
        assert thread_args[1] == {"test.py": "item1"}
        assert thread_args[2]["deep"] is True

def test_run_rescan_updates_ui():
    paths = ["test.py"]
    item_map = {"test.py": "item1"}
    settings = {"deep": True, "gpt": False, "dry": False}
    cancel_event = threading.Event()

    # Mock scan_files to yield a result
    mock_result_data = ("test.py", "80%", "Admin", "User", "90%", "New Snippet")
    mock_events = [
        ("progress", (1, 1, "Testing")),
        ("result", mock_result_data),
        ("summary", (1, 100, 0.5))
    ]
    with patch("gptscan.scan_files", return_value=mock_events), \
         patch("gptscan.enqueue_ui_update") as mock_enqueue, \
         patch("gptscan.get_effective_confidence", return_value=80.0):

        gptscan.run_rescan(paths, item_map, settings, cancel_event)

        # Check if update_tree_row was enqueued
        mock_enqueue.assert_any_call(gptscan.update_tree_row, "item1", mock_result_data)
        # Check if finish_scan_state was enqueued
        mock_enqueue.assert_any_call(gptscan.finish_scan_state, 1, 1, 100, 0.5)

def test_update_tree_row():
    gptscan.tree = MagicMock()
    gptscan.tree.exists.return_value = True

    with patch("gptscan._prepare_tree_row", return_value=(["wrapped"], ("tag",))):
        gptscan.update_tree_row("item1", ("test.py", "80%", "", "", "", "snippet"))
        gptscan.tree.item.assert_called_with("item1", values=["wrapped"], tags=("tag",))

def test_rescan_no_selection():
    gptscan.tree = MagicMock()
    gptscan.tree.selection.return_value = ()

    with patch("threading.Thread") as mock_thread:
        gptscan.rescan_selected()
        mock_thread.assert_not_called()
