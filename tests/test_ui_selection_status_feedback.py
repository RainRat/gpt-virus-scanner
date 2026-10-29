import os
import unittest
from unittest.mock import MagicMock, patch
import gptscan


class TestUISelectionStatusFeedback(unittest.TestCase):
    """Test suite for Treeview selection status bar feedback in update_button_states."""

    def setUp(self):
        self.mock_tree = MagicMock()
        self.mock_status_label = MagicMock()

        self.patcher_tree = patch.object(gptscan, 'tree', self.mock_tree)
        self.patcher_status = patch.object(gptscan, 'status_label', self.mock_status_label)
        self.patcher_cancel = patch.object(gptscan, 'current_cancel_event', None)
        self.patcher_cache = patch.object(gptscan, '_all_results_cache', [])
        self.patcher_last_summary = patch.object(gptscan, '_last_scan_summary', "Scan complete: 10 files scanned, 2 suspicious files found.")

        self.patcher_tree.start()
        self.patcher_status.start()
        self.patcher_cancel.start()
        self.patcher_cache.start()
        self.patcher_last_summary.start()

    def tearDown(self):
        self.patcher_tree.stop()
        self.patcher_status.stop()
        self.patcher_cancel.stop()
        self.patcher_cache.stop()
        self.patcher_last_summary.stop()

    def test_single_selection_status_feedback(self):
        """Test status bar text when a single Treeview item is selected."""
        self.mock_tree.selection.return_value = ["item1"]

        sample_data = ["/path/to/suspicious.py", "85%", "Admin Note", "User Note", "85%", "eval(input)", "12"]

        with patch.object(gptscan, '_get_item_raw_values', return_value=sample_data):
            gptscan.update_button_states()

        expected_msg = "Selected 1 item (suspicious.py - 85% Threat)"
        self.mock_status_label.config.assert_called_with(text=expected_msg)

    def test_multiple_selection_status_feedback(self):
        """Test status bar text when multiple Treeview items are selected."""
        self.mock_tree.selection.return_value = ["item1", "item2"]

        item1_data = ["/path/to/high_threat.py", "90%", "Admin", "User", "90%", "os.system('rm -rf /')", "5"]
        item2_data = ["/path/to/medium_threat.py", "60%", "Admin", "User", "60%", "exec(code)", "20"]

        def mock_get_item(item_id):
            if item_id == "item1":
                return item1_data
            if item_id == "item2":
                return item2_data
            return None

        with patch.object(gptscan, '_get_item_raw_values', side_effect=mock_get_item):
            gptscan.update_button_states()

        expected_msg = "Selected 2 items (1 high risk, 1 medium risk)"
        self.mock_status_label.config.assert_called_with(text=expected_msg)

    def test_empty_selection_restores_status(self):
        """Test status bar restoration when selection is cleared."""
        self.mock_tree.selection.return_value = []
        self.mock_tree.get_children.return_value = []

        mock_filter_var = MagicMock()
        mock_filter_var.get.return_value = ""

        with patch.object(gptscan, 'filter_var', mock_filter_var):
            gptscan.update_button_states()

        self.mock_status_label.config.assert_called_with(text="Scan complete: 10 files scanned, 2 suspicious files found.")


if __name__ == '__main__':
    unittest.main()
