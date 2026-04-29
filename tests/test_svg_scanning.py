import pytest
from gptscan import unpack_content, Config

def test_is_container_svg():
    """Verify that .svg files are recognized as containers."""
    assert Config.is_container("test.svg") is True
    assert Config.is_container("path/to/test.svg") is True

def test_unpack_content_svg_extraction():
    """Verify script and attribute extraction from SVG files."""
    svg_content = """
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <script>
            alert('SVG Script');
        </script>
        <circle cx="50" cy="50" r="40" onclick="console.log('clicked')" onload="malicious()"/>
        <a href="javascript:void(0)">
            <rect width="10" height="10" />
        </a>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("test.svg", content_bytes))

    # Should find Script 1 and Attributes
    assert len(snippets) == 2

    names = [s[0] for s in snippets]
    assert "test.svg [Script 1]" in names
    assert "test.svg [Attributes]" in names

    scripts = {s[0]: s[1].decode('utf-8') for s in snippets}

    assert "alert('SVG Script');" in scripts["test.svg [Script 1]"]

    attr_text = scripts["test.svg [Attributes]"]
    assert "// onclick" in attr_text
    assert "console.log('clicked')" in attr_text
    assert "// onload" in attr_text
    assert "malicious()" in attr_text
    assert "// href" in attr_text
    assert "javascript:void(0)" in attr_text

def test_unpack_content_svg_entities():
    """Verify that HTML entities in SVG attributes are unescaped."""
    svg_content = '<circle onclick="alert(&quot;hit&quot;)"/>'
    snippets = list(unpack_content("entities.svg", svg_content.encode('utf-8')))

    assert len(snippets) == 1
    assert 'alert("hit")' in snippets[0][1].decode('utf-8')

def test_unpack_content_svg_no_scripts():
    """Verify that SVG files with no scripts yield the original file as fallback."""
    svg_content = '<svg><circle /></svg>'
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("noscripts.svg", content_bytes))

    assert len(snippets) == 1
    assert snippets[0] == ("noscripts.svg", content_bytes)
