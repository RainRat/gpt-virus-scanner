import pytest
import gptscan
from gptscan import unpack_content

def test_unpack_content_html_extraction():
    """Verify that <script> blocks are correctly extracted from HTML files."""
    html_content = """
    <html>
        <head>
            <script>
                console.log('Script 1');
            </script>
        </head>
        <body>
            <h1>Test</h1>
            <script type="text/javascript">
                const x = 10;
                alert(x);
            </script>
            <script>
                // Empty script should be ignored if strip() is empty
            </script>
            <script>    </script>
            <p>More text</p>
            <SCRIPT>
                console.log('Case insensitive');
            </SCRIPT>
        </body>
    </html>
    """
    content_bytes = html_content.encode('utf-8')

    # Test with .html extension
    snippets = list(unpack_content("test.html", content_bytes))

    assert len(snippets) == 4

    assert snippets[0][0] == "test.html [Script 1]"
    assert "console.log('Script 1');" in snippets[0][1].decode('utf-8')

    assert snippets[1][0] == "test.html [Script 2]"
    assert "const x = 10;" in snippets[1][1].decode('utf-8')
    assert "alert(x);" in snippets[1][1].decode('utf-8')

    assert snippets[2][0] == "test.html [Script 3]"
    assert "// Empty script" in snippets[2][1].decode('utf-8')

    assert snippets[3][0] == "test.html [Script 5]"
    assert "console.log('Case insensitive');" in snippets[3][1].decode('utf-8')

def test_unpack_content_html_no_scripts():
    """Verify that HTML files with no scripts or embedded elements yield the original file (fallback)."""
    html_content = "<html><body><h1>No Scripts</h1></body></html>"
    content_bytes = html_content.encode('utf-8')

    snippets = list(unpack_content("noscripts.html", content_bytes))
    # It should fall back to yielding the original file because it's a supported extension
    assert len(snippets) == 1
    assert snippets[0] == ("noscripts.html", content_bytes)

def test_unpack_content_xhtml_extraction():
    """Verify that <script> blocks are correctly extracted from XHTML files."""
    xhtml_content = """
    <html xmlns="http://www.w3.org/1999/xhtml">
        <script>alert('XHTML');</script>
    </html>
    """
    snippets = list(unpack_content("test.xhtml", xhtml_content.encode('utf-8')))
    assert len(snippets) == 1
    assert snippets[0][0] == "test.xhtml [Script 1]"
    assert b"alert('XHTML');" in snippets[0][1]

def test_unpack_content_html_embedded_elements():
    """Verify that iframes, objects, embeds, and applets are correctly extracted and bundled."""
    html_content = """
    <html>
        <body>
            <script>alert('script');</script>
            <iframe src="http://malicious-iframe.com"></iframe>
            <object data="malicious.swf"></object>
            <embed src="malware.exe">
            <applet code="Malicious.class"></applet>
            <iframe src="no-closing-tag">
        </body>
    </html>
    """
    content_bytes = html_content.encode('utf-8')
    snippets = list(unpack_content("test.html", content_bytes))

    # Should have Script 1 and Embedded Elements
    assert len(snippets) == 2
    assert snippets[0][0] == "test.html [Script 1]"
    assert snippets[1][0] == "test.html [Embedded Elements]"
    
    embedded_text = snippets[1][1].decode('utf-8')
    assert '<iframe src="http://malicious-iframe.com"></iframe>' in embedded_text
    assert '<object data="malicious.swf"></object>' in embedded_text
    assert '<embed src="malware.exe">' in embedded_text
    assert '<applet code="Malicious.class"></applet>' in embedded_text
    assert '<iframe src="no-closing-tag">' in embedded_text

