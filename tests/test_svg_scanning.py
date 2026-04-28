import pytest
import os
from gptscan import unpack_content, Config

def test_svg_is_container():
    assert Config.is_container("test.svg") is True
    assert Config.is_container("test.SVG") is True

def test_svg_script_extraction():
    svg_content = b"""
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <script type="text/javascript">
        console.log("Malicious SVG script");
      </script>
      <circle cx="50" cy="50" r="40" onclick="alert('clicked')" />
      <a href="javascript:alert('link')">Click me</a>
    </svg>
    """
    snippets = list(unpack_content("test.svg", svg_content))

    # Check for <script> block
    assert any("test.svg [Script 1]" in s[0] for s in snippets)
    assert any(b"Malicious SVG script" in s[1] for s in snippets)

    # Check for Inline Scripts bundle
    assert any("test.svg [Inline Scripts]" in s[0] for s in snippets)
    assert any(b"onclick: alert('clicked')" in s[1] for s in snippets)
    assert any(b"javascript:alert('link')" in s[1] for s in snippets)

def test_html_inline_script_extraction():
    html_content = b"""
    <html>
      <body onload="doSomething()">
        <button onclick="runPayload()">Click</button>
        <a href="javascript:void(0)">Link</a>
      </body>
    </html>
    """
    snippets = list(unpack_content("test.html", html_content))

    assert any("test.html [Inline Scripts]" in s[0] for s in snippets)
    assert any(b"onload: doSomething()" in s[1] for s in snippets)
    assert any(b"onclick: runPayload()" in s[1] for s in snippets)
    assert any(b"javascript:void(0)" in s[1] for s in snippets)
