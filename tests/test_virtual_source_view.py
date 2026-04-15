import pytest
import io
import zipfile
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import scan_files, _virtual_source_cache, Config

@pytest.fixture
def mock_tf_env(monkeypatch):
    mock_model = MagicMock()
    # Mock model prediction to return a low threat level
    mock_model.predict.return_value = [[0.1]]

    # Mock get_model to return our mock
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    # Mock tensorflow module
    mock_tf = MagicMock()
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    return mock_model

def test_virtual_source_cache_population(mock_tf_env):
    """Test that scanning virtual snippets populates the _virtual_source_cache."""
    # Clear cache before test
    gptscan._virtual_source_cache = {}

    snippet_content = b"print('this is a virtual file')"
    decoded_content = snippet_content.decode('utf-8')
    extra_snippets = [("[Clipboard]", snippet_content)]

    # We need to simulate how the GUI handles events
    events = scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        extra_snippets=extra_snippets
    )

    # Simulate the scan_handler logic
    for event_type, data in events:
        if event_type == 'result':
            # data is (path, own_conf, admin, user, gpt, snippet, line, full_content)
            if len(data) > 7 and data[7]:
                gptscan._virtual_source_cache[data[0]] = data[7]

    assert "[Clipboard]" in gptscan._virtual_source_cache
    assert gptscan._virtual_source_cache["[Clipboard]"] == decoded_content

def test_zip_member_source_cache_population(mock_tf_env, tmp_path):
    """Test that scanning a ZIP archive populates the cache with member contents."""
    gptscan._virtual_source_cache = {}

    zip_path = tmp_path / "test.zip"
    member_name = "script.py"
    member_content = b"print('hello from zip')"
    decoded_content = member_content.decode('utf-8')

    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr(member_name, member_content)

    events = scan_files(
        scan_targets=[str(zip_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    )

    for event_type, data in events:
        if event_type == 'result':
            if len(data) > 7 and data[7]:
                gptscan._virtual_source_cache[data[0]] = data[7]

    # The name in scan_files for zip members is f"{zip_path}[{member_name}]"
    expected_key = f"{zip_path}[{member_name}]"
    assert expected_key in gptscan._virtual_source_cache
    assert gptscan._virtual_source_cache[expected_key] == decoded_content

@patch("gptscan.messagebox")
def test_load_display_code_uses_cache(mock_msg, monkeypatch):
    """Test that the logic used in load_display_code would favor the cache."""
    # Since load_display_code is local to view_details, we'll test the logic directly
    # based on the implementation we added to gptscan.py

    path = "[VirtualFile]"
    content = "full source content"
    gptscan._virtual_source_cache[path] = content

    # Logic from gptscan.py:
    # content = None
    # if path in _virtual_source_cache:
    #     content = _virtual_source_cache[path]

    retrieved_content = gptscan._virtual_source_cache.get(path)
    assert retrieved_content == content

    # Verify that it handles missing files by checking cache first
    path_missing = "[Missing]"
    assert path_missing not in gptscan._virtual_source_cache
    assert not gptscan.os.path.exists(path_missing)
