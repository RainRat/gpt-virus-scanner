import hashlib
import pytest
from gptscan import get_virustotal_url

def test_get_virustotal_url_local_file(tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"

    assert get_virustotal_url(str(f)) == expected_url

def test_get_virustotal_url_non_existent_file():
    assert get_virustotal_url("non_existent_file.py") is None

def test_get_virustotal_url_virtual_path_with_snippet():
    snippet = "print('virtual')"
    expected_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()
    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"

    assert get_virustotal_url("[Clipboard]", snippet=snippet) == expected_url

def test_get_virustotal_url_virtual_path_without_snippet():
    assert get_virustotal_url("[Stdin]") is None
