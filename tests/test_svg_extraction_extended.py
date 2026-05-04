import pytest
from gptscan import unpack_content

def test_unpack_content_svg_extended_extraction():
    """Verify that multiple scripts and various attributes are extracted from SVG."""
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <script>
            console.log('SVG Script 1');
        </script>
        <circle cx="50" cy="50" r="40" onclick="alert('circle clicked')" onmouseover="console.log('hover')"/>
        <script type="text/ecmascript">
            console.log('SVG Script 2');
        </script>
        <rect width="100" height="100" fill="red">
            <set attributeName="onmouseover" to="alert('rect hover')"/>
        </rect>
        <a xlink:href="javascript:void(0)" href="javascript:alert('link')">
            <text x="10" y="20">Click me</text>
        </a>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("test.svg", content_bytes))

    # Expect [Script 1], [Script 2], and [Attributes]
    names = [s[0] for s in snippets]
    assert "test.svg [Script 1]" in names
    assert "test.svg [Script 2]" in names
    assert "test.svg [Attributes]" in names

    scripts = {s[0]: s[1].decode('utf-8') for s in snippets}
    assert "console.log('SVG Script 1');" in scripts["test.svg [Script 1]"]
    assert "console.log('SVG Script 2');" in scripts["test.svg [Script 2]"]

    attr_text = scripts["test.svg [Attributes]"]
    assert "alert('circle clicked')" in attr_text
    assert "console.log('hover')" in attr_text
    # Note: current regex might not catch xlink:href if it only looks for href/src
    # but it should catch href="javascript:alert('link')"
    assert "javascript:alert('link')" in attr_text

def test_unpack_content_svg_entities_extraction():
    """Verify that HTML entities in SVG attributes are unescaped."""
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg">
        <rect onclick="alert(&apos;entities&apos;)" onmouseover="eval(&quot;evil()&quot;)"/>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("entities.svg", content_bytes))

    attr_text = [s for s in snippets if "[Attributes]" in s[0]][0][1].decode('utf-8')
    assert "alert('entities')" in attr_text
    assert 'eval("evil()")' in attr_text

def test_unpack_content_svg_no_scripts():
    """Verify that SVG without scripts yields the original file."""
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="5"/></svg>"""
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("noscripts.svg", content_bytes))

    assert len(snippets) == 1
    assert snippets[0] == ("noscripts.svg", content_bytes)
