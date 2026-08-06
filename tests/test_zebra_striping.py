import pytest
from unittest.mock import MagicMock
import gptscan

def test_zebra_striping_insert_tree_row(monkeypatch):
    monkeypatch.setattr(gptscan.Config, 'THRESHOLD', 50)
    mock_all_var = MagicMock()
    mock_all_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'all_var', mock_all_var)

    # Mock tree
    mock_tree = MagicMock()
    mock_tree.column.return_value = {'width': 100}
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'default_font_measure', lambda x: 10)

    # We clear the cached results before testing
    monkeypatch.setattr(gptscan, '_all_results_cache', [])

    # 1. When current children count is 0 (even, index 0): should NOT have 'odd' tag
    mock_tree.get_children.return_value = []
    gptscan.insert_tree_row(("safe_0.py", "10%", "admin", "user", "0%", "snippet"))
    _, kwargs = mock_tree.insert.call_args
    assert 'odd' not in kwargs['tags']

    # 2. When current children count is 1 (odd, index 1): should have 'odd' tag
    # Since we mock get_children to return a list with 1 item, len() is 1, so is_odd is True!
    mock_tree.get_children.return_value = ["item_0"]
    gptscan.insert_tree_row(("safe_1.py", "10%", "admin", "user", "0%", "snippet"))
    _, kwargs = mock_tree.insert.call_args
    assert 'odd' in kwargs['tags']

    # 3. When current children count is 2 (even, index 2): should NOT have 'odd' tag
    mock_tree.get_children.return_value = ["item_0", "item_1"]
    gptscan.insert_tree_row(("safe_2.py", "10%", "admin", "user", "0%", "snippet"))
    _, kwargs = mock_tree.insert.call_args
    assert 'odd' not in kwargs['tags']

def test_zebra_striping_apply_filter(monkeypatch):
    monkeypatch.setattr(gptscan.Config, 'THRESHOLD', 50)
    mock_all_var = MagicMock()
    mock_all_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'all_var', mock_all_var)

    mock_tree = MagicMock()
    mock_tree.column.return_value = {'width': 100}
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'default_font_measure', lambda x: 10)

    # Set cache with multiple safe files
    cache = [
        ("file0.py", "10%", "", "", "", "snippet0"),
        ("file1.py", "10%", "", "", "", "snippet1"),
        ("file2.py", "10%", "", "", "", "snippet2")
    ]
    monkeypatch.setattr(gptscan, '_all_results_cache', cache)

    # Run _apply_filter
    gptscan._apply_filter()

    # Check all calls to tree.insert
    assert mock_tree.insert.call_count == 3

    # First insert (index 0, even) should not have 'odd'
    args0, kwargs0 = mock_tree.insert.call_args_list[0]
    assert 'odd' not in kwargs0['tags']

    # Second insert (index 1, odd) should have 'odd'
    args1, kwargs1 = mock_tree.insert.call_args_list[1]
    assert 'odd' in kwargs1['tags']

    # Third insert (index 2, even) should not have 'odd'
    args2, kwargs2 = mock_tree.insert.call_args_list[2]
    assert 'odd' not in kwargs2['tags']

def test_zebra_striping_update_tree_row(monkeypatch):
    monkeypatch.setattr(gptscan.Config, 'THRESHOLD', 50)
    mock_all_var = MagicMock()
    mock_all_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'all_var', mock_all_var)

    mock_tree = MagicMock()
    mock_tree.column.return_value = {'width': 100}
    mock_tree.exists.return_value = True
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'default_font_measure', lambda x: 10)

    cache = [
        ("file0.py", "10%", "", "", "", "snippet0", 1),
        ("file1.py", "10%", "", "", "", "snippet1", 1),
        ("file2.py", "10%", "", "", "", "snippet2", 1)
    ]
    monkeypatch.setattr(gptscan, '_all_results_cache', cache)

    # 1. Update first row (index 0 in tree.get_children())
    mock_tree.get_children.return_value = ["item0", "item1", "item2"]
    gptscan.update_tree_row("item0", ("file0.py", "10%", "", "", "", "snippet0_updated", 1))
    _, kwargs0 = mock_tree.item.call_args
    assert 'odd' not in kwargs0['tags']

    # 2. Update second row (index 1 in tree.get_children())
    gptscan.update_tree_row("item1", ("file1.py", "10%", "", "", "", "snippet1_updated", 1))
    _, kwargs1 = mock_tree.item.call_args
    assert 'odd' in kwargs1['tags']
