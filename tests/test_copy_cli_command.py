import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import Config, deep_var, git_var, dry_var, all_var, gpt_var, textbox

@pytest.fixture
def mock_gui_vars():
    """Mock the Tkinter variables used by copy_cli_command."""
    with patch('gptscan.deep_var', MagicMock()) as mock_deep, \
         patch('gptscan.git_var', MagicMock()) as mock_git, \
         patch('gptscan.dry_var', MagicMock()) as mock_dry, \
         patch('gptscan.all_var', MagicMock()) as mock_all, \
         patch('gptscan.gpt_var', MagicMock()) as mock_gpt, \
         patch('gptscan.textbox', MagicMock()) as mock_textbox:

        # Default mock values
        mock_deep.get.return_value = False
        mock_git.get.return_value = False
        mock_dry.get.return_value = False
        mock_all.get.return_value = False
        mock_gpt.get.return_value = False
        mock_textbox.get.return_value = "/test/path"

        yield {
            'deep': mock_deep,
            'git': mock_git,
            'dry': mock_dry,
            'all': mock_all,
            'gpt': mock_gpt,
            'textbox': mock_textbox
        }

@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config settings between tests."""
    orig_threshold = Config.THRESHOLD
    orig_provider = Config.provider
    orig_model = Config.model_name
    orig_api_base = Config.api_base
    orig_last_path = Config.last_path

    yield

    Config.THRESHOLD = orig_threshold
    Config.provider = orig_provider
    Config.model_name = orig_model
    Config.api_base = orig_api_base
    Config.last_path = orig_last_path

def test_copy_cli_command_basic(mock_gui_vars):
    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        mock_root.clipboard_append.assert_called_once_with("python gptscan.py /test/path --cli")

def test_copy_cli_command_with_spaces_in_path(mock_gui_vars):
    # In multi-target mode, paths with spaces must be quoted in the textbox
    mock_gui_vars['textbox'].get.return_value = "'/path with spaces/script.py'"
    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        assert "python gptscan.py '/path with spaces/script.py' --cli" in mock_root.clipboard_append.call_args[0][0]

def test_copy_cli_command_all_options(mock_gui_vars):
    mock_gui_vars['deep'].get.return_value = True
    mock_gui_vars['git'].get.return_value = True
    mock_gui_vars['dry'].get.return_value = True
    mock_gui_vars['all'].get.return_value = True
    Config.THRESHOLD = 75

    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        command = mock_root.clipboard_append.call_args[0][0]
        assert "--deep" in command
        assert "--git-changes" in command
        assert "--dry-run" in command
        assert "--show-all" in command
        assert "--threshold 75" in command

def test_copy_cli_command_ai_options(mock_gui_vars):
    mock_gui_vars['gpt'].get.return_value = True
    Config.provider = "ollama"
    Config.model_name = "llama3.2"
    Config.api_base = "http://localhost:11434/v1"

    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        command = mock_root.clipboard_append.call_args[0][0]
        assert "--use-gpt" in command
        assert "--provider ollama" in command
        assert "--model llama3.2" in command
        assert "--api-base http://localhost:11434/v1" in command

def test_copy_cli_command_default_provider_omitted(mock_gui_vars):
    mock_gui_vars['gpt'].get.return_value = True
    Config.provider = "openai"

    with patch('gptscan.root', MagicMock()) as mock_root, \
         patch('gptscan.update_status'):
        gptscan.copy_cli_command()
        command = mock_root.clipboard_append.call_args[0][0]
        assert "--use-gpt" in command
        assert "--provider" not in command

def test_copy_cli_command_ui_update(mock_gui_vars):
    with patch('gptscan.root', MagicMock()), \
         patch('gptscan.update_status') as mock_update_status:

        gptscan.copy_cli_command()
        mock_update_status.assert_called_once_with("CLI command copied to clipboard.")
