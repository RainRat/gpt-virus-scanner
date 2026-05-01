
import unittest
from unittest.mock import MagicMock, patch
import gptscan

class TestThresholdBug(unittest.TestCase):
    @patch('gptscan.get_model')
    @patch('gptscan._tf_module')
    def test_threshold_logic(self, mock_tf, mock_get_model):
        # Mock model to return a low confidence (0.3)
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.3]]
        mock_get_model.return_value = mock_model

        # Mock tensorflow functions
        mock_tf.expand_dims.return_value = None
        mock_tf.constant.return_value = None

        # Set Config THRESHOLD to 50 (0.5)
        gptscan.Config.THRESHOLD = 50

        # We want to scan a file and see if it yields the best result even if below threshold
        # when show_all is False.

        # Create a dummy file
        with open("dummy.txt", "w") as f:
            f.write("safe content")

        # Run scan_files
        results = list(gptscan.scan_files(
            scan_targets=["dummy.txt"],
            deep_scan=False,
            show_all=False,
            use_gpt=False
        ))

        # Extract 'result' events
        scan_results = [r for r in results if r[0] == 'result']

        # According to help text, we should have 1 result (the most suspicious one)
        # But currently it will be 0.
        print(f"Number of results found: {len(scan_results)}")
        for r in scan_results:
            print(f"Result: {r[1][0]} at {r[1][1]}")

        self.assertEqual(len(scan_results), 1, "Should have found the most suspicious result even if below threshold")

if __name__ == "__main__":
    unittest.main()
