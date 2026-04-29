import pytest
import gptscan
from gptscan import unpack_content, Config

def test_is_container_svg():
    """Check if .svg is recognized as a container."""
    assert Config.is_container("test.svg") is True

def test_unpack_content_svg_script():
    """Verify script extraction from SVG."""
    svg_content = """<?xml version="1.0" standalone="no"?>
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <script type="text/javascript">
        alert("SVG Script");
      </script>
      <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
    </svg>
    """
    snippets = list(unpack_content("test.svg", svg_content.encode('utf-8')))
    # Should find at least one Script snippet
    script_snippets = [s for s in snippets if "[Script" in s[0]]
    assert len(script_snippets) > 0
    assert "alert(\"SVG Script\");" in script_snippets[0][1].decode('utf-8')

def test_unpack_content_event_handlers():
    """Verify event handler extraction from HTML/SVG."""
    html_content = """
    <div onclick="alert('clicked')" onmouseover="console.log('hover')">
        <img src="x" onerror="confirm('error')">
    </div>
    """
    snippets = list(unpack_content("test.html", html_content.encode('utf-8')))
    # Should find "Embedded Scripts" or similar snippet for attributes
    attr_snippets = [s for s in snippets if "[Embedded Scripts]" in s[0]]
    assert len(attr_snippets) > 0

    content = attr_snippets[0][1].decode('utf-8')
    assert "alert('clicked')" in content
    assert "console.log('hover')" in content
    assert "confirm('error')" in content

def test_unpack_content_javascript_urls():
    """Verify javascript: URL extraction."""
    html_content = '<a href="javascript:alert(1)">Click me</a>'
    snippets = list(unpack_content("test.html", html_content.encode('utf-8')))

    attr_snippets = [s for s in snippets if "[Embedded Scripts]" in s[0]]
    assert len(attr_snippets) > 0
    assert "alert(1)" in attr_snippets[0][1].decode('utf-8')

def test_unpack_content_html_entities():
    """Verify unescaping of HTML entities in attributes."""
    html_content = '<img src="x" onerror="alert(&quot;unescaped&quot;)">'
    snippets = list(unpack_content("test.html", html_content.encode('utf-8')))

    attr_snippets = [s for s in snippets if "[Embedded Scripts]" in s[0]]
    assert len(attr_snippets) > 0
    assert 'alert("unescaped")' in attr_snippets[0][1].decode('utf-8')
