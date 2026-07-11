import pytest
from gptscan import get_online_url
from unittest.mock import patch

def test_get_online_url_remote_restoration():
    """Verify that [URL] targets are correctly restored from raw to view links."""
    # GitHub: already tested in test_view_online.py but good to keep here
    assert get_online_url("[URL] https://raw.githubusercontent.com/u/r/m/f.py", 10) == "https://github.com/u/r/blob/m/f.py#L10"

    # GitLab
    assert get_online_url("[URL] https://gitlab.com/u/r/-/raw/m/f.py", 10) == "https://gitlab.com/u/r/-/blob/m/f.py#L10"

    # Bitbucket
    assert get_online_url("[URL] https://bitbucket.org/u/r/raw/m/f.py", 10) == "https://bitbucket.org/u/r/src/m/f.py#lines-10"

    # Pastebin
    assert get_online_url("[URL] https://pastebin.com/raw/abcdef", 1) == "https://pastebin.com/abcdef#L1"

    # Hugging Face
    assert get_online_url("[URL] https://huggingface.co/u/r/raw/m/f.py", 10) == "https://huggingface.co/u/r/blob/m/f.py#L10"

@patch('gptscan._get_git_info')
@patch('gptscan.subprocess.check_output')
def test_get_online_url_ssh_normalization(mock_check_output, mock_info):
    """Verify normalization of various SSH remote formats."""
    mock_info.return_value = ("/app", "file.py")

    # Test cases: (remote_url, expected_base)
    test_cases = [
        ("git@github.com:user/repo.git", "https://github.com/user/repo"),
        ("ssh://git@github.com/user/repo.git", "https://github.com/user/repo"),
        ("ssh://git@gitlab.com:group/sub/repo.git", "https://gitlab.com/group/sub/repo"),
        ("git@bitbucket.org:user/repo", "https://bitbucket.org/user/repo"),
    ]

    for remote, expected_base in test_cases:
        def side_effect(cmd, **kwargs):
            if cmd == ["git", "remote", "get-url", "origin"]:
                return remote
            if cmd == ["git", "rev-parse", "HEAD"]:
                return "rev"
            return ""
        mock_check_output.side_effect = side_effect

        url = get_online_url("/app/file.py", 5)

        if "github.com" in expected_base:
            assert url == f"{expected_base}/blob/rev/file.py#L5"
        elif "gitlab.com" in expected_base:
            assert url == f"{expected_base}/-/blob/rev/file.py#L5"
        elif "bitbucket.org" in expected_base:
            assert url == f"{expected_base}/src/rev/file.py#lines-5"

def test_get_online_url_no_line():
    """Verify behavior when line is None or 0."""
    assert get_online_url("[URL] https://github.com/u/r/blob/m/f.py", None) == "https://github.com/u/r/blob/m/f.py"
    assert get_online_url("[URL] https://github.com/u/r/blob/m/f.py", 0) == "https://github.com/u/r/blob/m/f.py"
    assert get_online_url("[URL] https://github.com/u/r/blob/m/f.py", "abc") == "https://github.com/u/r/blob/m/f.py"
