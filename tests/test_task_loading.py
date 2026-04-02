import pytest
from gptscan import Config, load_file

@pytest.fixture
def temp_task_file(tmp_path):
    """Create a multi-line task.txt for testing."""
    task_file = tmp_path / "task.txt"
    content = "Line 1\nLine 2\nLine 3"
    task_file.write_text(content)
    return task_file

def test_load_file_modes(temp_task_file):
    """Directly test load_file behavior with the new 'full' mode."""
    # Single line mode (default) should return only the first line
    assert load_file(str(temp_task_file)) == "Line 1"

    # Multi line mode should return a list of lines
    assert load_file(str(temp_task_file), mode='multi_line') == ["Line 1", "Line 2", "Line 3"]

    # Full mode should return the entire content as a string
    assert load_file(str(temp_task_file), mode='full') == "Line 1\nLine 2\nLine 3"

def test_load_file_missing_full_mode():
    """Verify load_file returns empty string for missing file in 'full' mode."""
    assert load_file("nonexistent.txt", mode='full') == ""
