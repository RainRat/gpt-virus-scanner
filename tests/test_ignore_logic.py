import os
import pytest
from pathlib import Path
from gptscan import Config, add_to_ignore_file, remove_from_ignore_file

@pytest.fixture
def temp_ignore_setup(tmp_path, monkeypatch):
    """Set up a temporary environment for ignore logic tests."""
    ignore_file = tmp_path / ".gptscanignore"
    monkeypatch.chdir(tmp_path)

    # Backup and reset Config state
    original_patterns = Config.ignore_patterns
    Config.ignore_patterns = []

    yield ignore_file

    # Restore
    Config.ignore_patterns = original_patterns

def test_config_initialize_ignores_comments_and_empty_lines(temp_ignore_setup):
    """Verify that Config.initialize correctly parses .gptscanignore."""
    temp_ignore_setup.write_text(
        "pattern1 # trailing comment\n"
        "# full line comment\n"
        "  \n" # whitespace
        "pattern2\n"
        "  pattern3  # indented\n"
        "  # another full line\n"
        "pattern4"
    )

    Config.initialize()

    assert Config.ignore_patterns == ["pattern1", "pattern2", "pattern3", "pattern4"]
    assert "pattern1 # trailing comment" not in Config.ignore_patterns

def test_add_to_ignore_file_deduplication_with_comments(temp_ignore_setup):
    """Test that add_to_ignore_file prevents duplicates regardless of inline comments."""
    # Start with an existing file containing a commented pattern
    temp_ignore_setup.write_text("existing # first comment\n")
    Config.ignore_patterns = ["existing"]

    # Add the same pattern with a different comment
    add_to_ignore_file("existing # second comment")

    # File should remain unchanged, and patterns list should still have 1 item
    content = temp_ignore_setup.read_text()
    assert "existing # first comment" in content
    assert "existing # second comment" not in content
    assert Config.ignore_patterns == ["existing"]

    # Add a brand new pattern
    add_to_ignore_file("new # with tag")
    assert "new" in Config.ignore_patterns
    assert "new # with tag" in temp_ignore_setup.read_text()

def test_remove_from_ignore_file_preserves_unrelated_content(temp_ignore_setup):
    """Test that removal correctly identifies patterns even with comments and preserves others."""
    initial_content = (
        "p1 # c1\n"
        "# full comment\n"
        "\n"
        "p2\n"
        "p3 # c3"
    )
    temp_ignore_setup.write_text(initial_content)
    Config.ignore_patterns = ["p1", "p2", "p3"]

    # Remove p1 and p3
    remove_from_ignore_file(["p1", "p3"])

    new_content = temp_ignore_setup.read_text()

    # p1 and p3 should be gone
    assert "p1" not in new_content
    assert "p3" not in new_content

    # p2, full comment, and empty line should remain
    assert "p2" in new_content
    assert "# full comment" in new_content
    assert "\n\n" in new_content

    assert Config.ignore_patterns == ["p2"]

def test_add_to_ignore_file_handles_whitespace_only_with_comment(temp_ignore_setup):
    """Ensure that lines that would result in empty patterns are not added."""
    add_to_ignore_file("# just a comment")
    add_to_ignore_file("  ")

    assert Config.ignore_patterns == []
    if temp_ignore_setup.exists():
        assert temp_ignore_setup.read_text().strip() == ""

def test_remove_from_ignore_file_handles_non_existent(temp_ignore_setup):
    """Ensure removal doesn't crash if pattern or file doesn't exist."""
    # File doesn't exist yet
    remove_from_ignore_file(["ghost"]) # Should not crash

    temp_ignore_setup.write_text("p1\n")
    Config.ignore_patterns = ["p1"]

    remove_from_ignore_file(["ghost"]) # Should not change anything
    assert temp_ignore_setup.read_text() == "p1\n"
    assert Config.ignore_patterns == ["p1"]
