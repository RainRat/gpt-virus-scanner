import pytest
from gptscan import get_online_url
from unittest.mock import patch

def test_get_online_url_remote_target_restoration():
    # GitHub (Existing)
    url = get_online_url("[URL] https://raw.githubusercontent.com/user/repo/main/file.py", 10)
    assert url == "https://github.com/user/repo/blob/main/file.py#L10"

    # GitLab (Existing)
    url = get_online_url("[URL] https://gitlab.com/user/repo/-/raw/main/file.py", 10)
    assert url == "https://gitlab.com/user/repo/-/blob/main/file.py#L10"

    # Bitbucket (New)
    url = get_online_url("[URL] https://bitbucket.org/user/repo/raw/main/file.py", 10)
    assert url == "https://bitbucket.org/user/repo/src/main/file.py#lines-10"

    # Pastebin (New)
    url = get_online_url("[URL] https://pastebin.com/raw/abcdefgh", 1)
    assert url == "https://pastebin.com/abcdefgh#L1"

    # Hugging Face (New)
    url = get_online_url("[URL] https://huggingface.co/user/repo/raw/main/model.py", 5)
    assert url == "https://huggingface.co/user/repo/blob/main/model.py#L5"

@patch('gptscan._get_git_info')
@patch('gptscan.subprocess.check_output')
def test_get_online_url_ssh_formats(mock_check_output, mock_info):
    mock_info.return_value = ("/app", "src/file.py")

    # Test ssh://git@github.com/user/repo.git
    def side_effect_ssh_protocol(cmd, **kwargs):
        if "remote" in cmd: return "ssh://git@github.com/user/repo.git"
        return "abcdef"

    mock_check_output.side_effect = side_effect_ssh_protocol
    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://github.com/user/repo/blob/abcdef/src/file.py#L10"

    # Test git@github.com:user/repo.git (Standard SSH)
    def side_effect_standard_ssh(cmd, **kwargs):
        if "remote" in cmd: return "git@github.com:user/repo.git"
        return "abcdef"

    mock_check_output.side_effect = side_effect_standard_ssh
    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://github.com/user/repo/blob/abcdef/src/file.py#L10"

@patch('gptscan._get_git_info')
@patch('gptscan.subprocess.check_output')
def test_get_online_url_local_bitbucket(mock_check_output, mock_info):
    mock_info.return_value = ("/app", "src/file.py")

    def side_effect(cmd, **kwargs):
        if "remote" in cmd: return "https://bitbucket.org/user/repo.git"
        return "main"

    mock_check_output.side_effect = side_effect
    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://bitbucket.org/user/repo/src/main/src/file.py#lines-10"
