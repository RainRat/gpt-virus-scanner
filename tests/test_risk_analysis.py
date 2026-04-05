import json
import pytest
from unittest.mock import MagicMock
from gptscan import get_risk_category, get_effective_threat_level, _prepare_tree_row, Config

@pytest.mark.parametrize("conf, threshold, expected", [
    (90.0, 50, 'high'),
    (80.0, 50, 'high'),
    (79.9, 50, 'medium'),
    (50.0, 50, 'medium'),
    (49.9, 50, None),
    (10.0, 50, None),
    (0.0, 50, None),
    (-1.0, 50, None),
    (90.0, 95, None),
    (85.0, 80, 'high'),
    (80.0, 80, 'high'),
    (79.0, 80, None),
    (100.0, 0, 'high'),
    (50.0, 0, 'medium'),
    (0.0, 0, 'medium'),
])
def test_get_risk_category(conf, threshold, expected):
    assert get_risk_category(conf, threshold) == expected

@pytest.mark.parametrize("own, gpt, expected", [
    ("50%", "80%", 80.0),        # GPT prioritized
    ("50%", "", 50.0),          # Fallback to local
    ("50%", "invalid", 50.0),     # Fallback to local on GPT error
    ("50%", "0%", 0.0),         # GPT prioritized even if 0
    ("", "80%", 80.0),          # GPT prioritized even if local missing
    ("", "", -1.0),             # Both missing
    ("invalid", "invalid", -1.0), # Both invalid
    ("Error", "", -1.0),        # Error string handling
    ("", "Error", -1.0),        # Error string handling
    ("Dry Run", "", -1.0),      # Dry Run string handling
    ("75%", "Error", 75.0),     # Fallback when GPT is Error
    ("Error", "90%", 90.0),     # GPT prioritized even if local is Error
])
def test_get_effective_threat_level(own, gpt, expected):
    assert get_effective_threat_level(own, gpt) == expected

def test_prepare_tree_row(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin", "user", "gpt_conf", "snippet", "line")
    mock_tree.column.return_value = {'width': 100}
    monkeypatch.setattr("gptscan.tree", mock_tree)
    monkeypatch.setattr(Config, "THRESHOLD", 50)
    monkeypatch.setattr("gptscan.default_font_measure", lambda x: 10)

    # Test High Risk
    values = ("high.py", "90%", "Admin Notes", "User Notes", "85%", "Snippet", 10)
    wrapped, tags = _prepare_tree_row(values)

    assert tags == ('high-risk',)
    assert wrapped[0] == "high.py"
    assert wrapped[4] == "85%"
    # 8th column (index 7) should be raw JSON of first 7 elements
    assert json.loads(wrapped[7]) == list(values)

    # Test Medium Risk
    values = ("medium.py", "60%", "Admin", "User", "", "Snippet", 1)
    wrapped, tags = _prepare_tree_row(values)
    assert tags == ('medium-risk',)

    # Test Safe
    values = ("safe.py", "10%", "", "", "", "Snippet", 1)
    wrapped, tags = _prepare_tree_row(values)
    assert tags == ()

    # Test boundary 80%
    values = ("boundary.py", "80%", "", "", "", "Snippet", 1)
    wrapped, tags = _prepare_tree_row(values)
    assert tags == ('high-risk',)

    # Test boundary Threshold
    values = ("threshold.py", "50%", "", "", "", "Snippet", 1)
    wrapped, tags = _prepare_tree_row(values)
    assert tags == ('medium-risk',)
