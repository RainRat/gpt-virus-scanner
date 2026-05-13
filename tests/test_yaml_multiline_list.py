import pytest
from gptscan import unpack_content

def test_yaml_multiline_block_in_list():
    """Verify that multi-line blocks using '|' inside a YAML list are correctly extracted."""
    yaml_content = b"""
steps:
  - name: Build
    run:
      - |
        npm install
        npm run build
      - echo "Done"
"""
    results = list(unpack_content("workflow.yml", yaml_content))

    # We expect 2 scripts extracted from the list under 'run'
    # Actually, current logic joins list items with \n if they are strings.
    # But if they are blocks, it might be different.

    scripts = [r[1].decode('utf-8') for r in results]

    # After my proposed fix, it should contain both
    assert any('npm install\nnpm run build' in s for s in scripts)
    assert any('echo "Done"' in s for s in scripts)

def test_yaml_folded_block_in_list():
    """Verify that folded blocks using '>' inside a YAML list are correctly extracted."""
    yaml_content = b"""
script:
  - >
    echo "This is a
    long command"
  - ls -la
"""
    results = list(unpack_content("test.yml", yaml_content))
    scripts = [r[1].decode('utf-8') for r in results]

    assert any('echo "This is a\nlong command"' in s for s in scripts)
    assert any('ls -la' in s for s in scripts)

def test_yaml_mixed_list_content():
    """Verify a list with mixed single-line and multi-line items."""
    yaml_content = b"""
run:
  - echo "start"
  - |
    line 1
    line 2
  - echo "end"
"""
    results = list(unpack_content("test.yml", yaml_content))
    scripts = [r[1].decode('utf-8') for r in results]

    assert 'echo "start"\nline 1\nline 2\necho "end"' in scripts or \
           (any('echo "start"' in s for s in scripts) and \
            any('line 1\nline 2' in s for s in scripts) and \
            any('echo "end"' in s for s in scripts))
    # Note: current implementation of list parsing in gptscan.py:
    # it accumulates list items and joins them with \n.
    # So we expect one snippet: 'echo "start"\nline 1\nline 2\necho "end"'
