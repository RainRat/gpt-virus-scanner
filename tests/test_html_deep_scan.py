import pytest
from gptscan import unpack_content, Config

def test_unpack_content_html_attribute_extraction():
    """Verify that inline event handlers and javascript: URLs are extracted from HTML."""
    html_content = """
    <html>
        <body>
            <button onclick="alert('clicked')">Click me</button>
            <img src="valid.png" onerror="eval(atob('bWFsY29kZSgp'))">
            <a href="javascript:console.log('link')">Link</a>
            <div onmouseover=" malicious_code() ">Hover</div>
            <a href='javascript:alert("single quotes")'>Single</a>
        </body>
    </html>
    """
    content_bytes = html_content.encode('utf-8')
    snippets = list(unpack_content("test.html", content_bytes))

    # Currently this will fail as these are not extracted yet
    # After implementation, we expect an [Attributes] snippet
    attr_snippets = [s for s in snippets if "[Attributes]" in s[0]]
    assert len(attr_snippets) == 1

    attr_text = attr_snippets[0][1].decode('utf-8')
    assert "alert('clicked')" in attr_text
    assert "eval(atob('bWFsY29kZSgp'))" in attr_text
    assert "console.log('link')" in attr_text
    assert "malicious_code()" in attr_text
    assert 'alert("single quotes")' in attr_text

def test_unpack_content_html_attribute_extraction_unquoted():
    """Verify that unquoted attributes and leading spaces in javascript: URLs are handled."""
    html_content = """
    <html>
        <body>
            <button onclick=alert(1)>Click</button>
            <a href="  javascript:alert(2)">Space</a>
            <a href=javascript:alert(3)>Unquoted JS</a>
        </body>
    </html>
    """
    snippets = list(unpack_content("test.html", html_content.encode('utf-8')))
    attr_text = [s for s in snippets if "[Attributes]" in s[0]][0][1].decode('utf-8')
    assert "alert(1)" in attr_text
    assert "javascript:alert(2)" in attr_text
    assert "javascript:alert(3)" in attr_text

def test_unpack_content_svg_extraction():
    """Verify that SVG files are treated as containers and scanned for scripts."""
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg">
        <script>alert('svg script')</script>
        <circle cx="50" cy="50" r="40" onclick="svg_evil()"/>
    </svg>
    """
    content_bytes = svg_content.encode('utf-8')

    # SVG must be in Config.is_container for this to work
    snippets = list(unpack_content("test.svg", content_bytes))

    assert any("[Script 1]" in s[0] for s in snippets)
    assert any("[Attributes]" in s[0] for s in snippets)

    script_snippet = [s for s in snippets if "[Script 1]" in s[0]][0][1].decode('utf-8')
    attr_snippet = [s for s in snippets if "[Attributes]" in s[0]][0][1].decode('utf-8')

    assert "alert('svg script')" in script_snippet
    assert "svg_evil()" in attr_snippet

def test_html_unescape_in_attributes():
    """Verify that HTML entities in attributes are unescaped before scanning."""
    html_content = '<button onclick="alert(&quot;hello&quot;)">Click</button>'
    content_bytes = html_content.encode('utf-8')
    snippets = list(unpack_content("test.html", content_bytes))

    attr_snippet = [s for s in snippets if "[Attributes]" in s[0]][0][1].decode('utf-8')
    assert 'alert("hello")' in attr_snippet
