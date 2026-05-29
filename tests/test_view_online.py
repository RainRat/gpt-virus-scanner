import pytest
from gptscan import get_online_url
from unittest.mock import patch, MagicMock

def test_get_online_url_remote_target():
    # GitHub blob
    url = get_online_url("[URL] https://github.com/user/repo/blob/main/file.py", 10)
    assert url == "https://github.com/user/repo/blob/main/file.py#L10"

    # GitHub raw
    url = get_online_url("[URL] https://raw.githubusercontent.com/user/repo/main/file.py", 10)
    assert url == "https://github.com/user/repo/blob/main/file.py#L10"

    # GitLab raw
    url = get_online_url("[URL] https://gitlab.com/user/repo/-/raw/main/file.py", 10)
    assert url == "https://gitlab.com/user/repo/-/blob/main/file.py#L10"

    # Bitbucket
    url = get_online_url("[URL] https://bitbucket.org/user/repo/src/main/file.py", 10)
    assert url == "https://bitbucket.org/user/repo/src/main/file.py#lines-10"

@patch('gptscan._get_git_info')
@patch('subprocess.check_output')
def test_get_online_url_local_file(mock_check_output, mock_info):
    mock_info.return_value = ("/app", "src/file.py")

    def side_effect(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "git@github.com:user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "abcdef"
        return ""

    mock_check_output.side_effect = side_effect

    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://github.com/user/repo/blob/abcdef/src/file.py#L10"

@patch('gptscan._get_git_info')
@patch('subprocess.check_output')
def test_get_online_url_local_file_gitlab(mock_check_output, mock_info):
    mock_info.return_value = ("/app", "src/file.py")

    def side_effect(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return "https://gitlab.com/user/repo.git"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "main"
        return ""

    mock_check_output.side_effect = side_effect

    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://gitlab.com/user/repo/-/blob/main/src/file.py#L10"

@patch('gptscan._get_git_info')
@patch('subprocess.check_output')
def test_get_online_url_unsupported_remote(mock_check_output, mock_info):
    mock_info.return_value = ("/app", "src/file.py")
    mock_check_output.return_value = "https://myserver.com/repo.git"

    url = get_online_url("/app/src/file.py", 10)
    assert url is None
