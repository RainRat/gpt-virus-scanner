import pytest
import io
from unittest.mock import MagicMock
import gptscan
from gptscan import scan_files, Config

def test_markdown_code_block_extraction(mock_tf_env, tmp_path):
    """Test that scripts inside Markdown code blocks are detected and scanned."""
    mock_tf_env.predict.side_effect = lambda tf_data, **kwargs: [[0.9]]
    md_path = tmp_path / "README.md"
    md_content = b"""
# Project README

Here is a python block:
```python
print('malicious block 1')
```

And a bash block:
```bash
echo 'malicious block 2'
```

Plain text here.
"""
    md_path.write_bytes(md_content)

    events = list(scan_files(
        scan_targets=[str(md_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']

    # Both code blocks should be found and scanned
    assert len(results) == 2

    # Check first block
    path1, own_conf1, _, _, _, snippet1, line1 = results[0]
    assert "README.md [Block 1]" in path1
    assert own_conf1 == "90%"
    assert "print('malicious block 1')" in snippet1

    # Check second block
    path2, own_conf2, _, _, _, snippet2, line2 = results[1]
    assert "README.md [Block 2]" in path2
    assert own_conf2 == "90%"
    assert "echo 'malicious block 2'" in snippet2

def test_markdown_no_blocks(mock_tf_env, tmp_path):
    """Test that Markdown files with no code blocks yield no results."""
    md_path = tmp_path / "safe.md"
    md_content = b"# Safe Markdown\nNo code here."
    md_path.write_bytes(md_content)

    events = list(scan_files(
        scan_targets=[str(md_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 0

def test_markdown_empty_blocks(mock_tf_env, tmp_path):
    """Test that empty code blocks are ignored."""
    md_path = tmp_path / "empty.md"
    md_content = b"```python\n\n```"
    md_path.write_bytes(md_content)

    events = list(scan_files(
        scan_targets=[str(md_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 0
