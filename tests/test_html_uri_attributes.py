import pytest
from gptscan import unpack_content

def test_html_uri_attributes_extraction():
    """Verify extraction of javascript: URIs from action, formaction, data, and xlink:href."""
    html_content = """
    <html>
        <body>
            <form action="javascript:alert('action')">
                <button formaction="javascript:alert('formaction')">Click</button>
            </form>
            <object data="javascript:alert('data')"></object>
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                <a xlink:href="javascript:alert('xlink')">link</a>
            </svg>
        </body>
    </html>
    """
    content_bytes = html_content.encode('utf-8')
    snippets = list(unpack_content("test.html", content_bytes))

    names = [s[0] for s in snippets]
    assert "test.html [Attributes]" in names

    attr_text = [s[1] for s in snippets if s[0] == "test.html [Attributes]"][0].decode('utf-8')
    assert "javascript:alert('action')" in attr_text
    assert "javascript:alert('formaction')" in attr_text
    assert "javascript:alert('data')" in attr_text
    assert "javascript:alert('xlink')" in attr_text

def test_svg_uri_attributes_extraction():
    """Verify extraction from SVG file specifically."""
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <a xlink:href="javascript:alert('xlink')">link</a>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')
    snippets = list(unpack_content("test.svg", content_bytes))

    names = [s[0] for s in snippets]
    assert "test.svg [Attributes]" in names

    attr_text = [s[1] for s in snippets if s[0] == "test.svg [Attributes]"][0].decode('utf-8')
    assert "javascript:alert('xlink')" in attr_text
