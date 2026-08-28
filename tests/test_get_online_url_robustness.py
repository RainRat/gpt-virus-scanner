import subprocess
from unittest.mock import patch
import pytest
from gptscan import get_online_url


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


def test_get_online_url_bracketed_non_url():
    """Verify paths starting with brackets that are not [URL] return None."""
    assert get_online_url("[ZIP] extract/file.py", 10) is None
    assert get_online_url("[ARCHIVE] test.py", 5) is None


@patch('gptscan._get_git_info')
def test_get_online_url_git_info_none(mock_info):
    """Verify get_online_url returns None when _get_git_info returns None."""
    mock_info.return_value = (None, None)
    assert get_online_url("/app/file.py", 10) is None


@patch('gptscan._get_git_info')
@patch('gptscan.subprocess.check_output')
def test_get_online_url_git_remote_failures(mock_check_output, mock_info):
    """Verify handling of git remote command errors or empty output."""
    mock_info.return_value = ("/app", "file.py")

    # Command raises CalledProcessError
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
    assert get_online_url("/app/file.py", 10) is None

    # Command raises OSError
    mock_check_output.side_effect = OSError("git not found")
    assert get_online_url("/app/file.py", 10) is None

    # Command returns empty string
    mock_check_output.side_effect = None
    mock_check_output.return_value = "   \n"
    assert get_online_url("/app/file.py", 10) is None


@patch('gptscan._get_git_info')
@patch('gptscan.subprocess.check_output')
def test_get_online_url_rev_parse_fallback_and_local_hosts(mock_check_output, mock_info):
    """Verify rev-parse fallback to HEAD on error/empty and local git resolution for Bitbucket/GitHub/GitLab without line numbers."""
    mock_info.return_value = ("/app", "src/file.py")

    # 1. rev-parse exception -> fallback rev="HEAD" for Bitbucket with line number
    def side_effect_err(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "https://bitbucket.org/user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            raise subprocess.CalledProcessError(1, "git")
        return ""
    mock_check_output.side_effect = side_effect_err
    assert get_online_url("/app/src/file.py", 15) == "https://bitbucket.org/user/repo/src/HEAD/src/file.py#lines-15"

    # 2. rev-parse empty string -> fallback rev="HEAD" for Bitbucket without line number
    def side_effect_empty(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "https://bitbucket.org/user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return ""
        return ""
    mock_check_output.side_effect = side_effect_empty
    assert get_online_url("/app/src/file.py", None) == "https://bitbucket.org/user/repo/src/HEAD/src/file.py"

    # 3. GitHub local file without line number
    def side_effect_gh(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "https://github.com/user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "commit123"
        return ""
    mock_check_output.side_effect = side_effect_gh
    assert get_online_url("/app/src/file.py", None) == "https://github.com/user/repo/blob/commit123/src/file.py"

    # 4. GitLab local file without line number
    def side_effect_gl(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "https://gitlab.com/user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "commit123"
        return ""
    mock_check_output.side_effect = side_effect_gl
    assert get_online_url("/app/src/file.py", None) == "https://gitlab.com/user/repo/-/blob/commit123/src/file.py"
