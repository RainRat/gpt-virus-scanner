import pytest
from gptscan import resolve_remote_url

def test_resolve_pastebin_url():
    # Test typical paste
    url = "https://pastebin.com/abcdefgh"
    # Expected behavior after implementation
    assert resolve_remote_url(url) == "https://pastebin.com/raw/abcdefgh"

def test_resolve_huggingface_url():
    # Test typical HF blob
    url = "https://huggingface.co/user/repo/blob/main/model.py"
    # Expected behavior after implementation
    assert resolve_remote_url(url) == "https://huggingface.co/user/repo/raw/main/model.py"

def test_pastebin_safe_urls():
    # These should NOT be transformed to /raw/
    safe_urls = [
        "https://pastebin.com/archive",
        "https://pastebin.com/tools",
        "https://pastebin.com/faq",
        "https://pastebin.com/contact",
        "https://pastebin.com/night_mode",
        "https://pastebin.com/pro",
        "https://pastebin.com/doc",
        "https://pastebin.com/signup",
        "https://pastebin.com/login",
        "https://pastebin.com/api",
        "https://pastebin.com/trends",
        "https://pastebin.com/languages",
    ]
    for url in safe_urls:
        assert resolve_remote_url(url) == url
