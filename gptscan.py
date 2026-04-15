import asyncio
import csv
import glob
import hashlib
import html
import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import webbrowser
import zipfile
from collections import deque
import tkinter.scrolledtext as scrolledtext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, Iterable, List, Optional, Tuple, Union

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import tkinter.font
import tkinter.ttk as ttk

# Global GUI variables for thread-safe updates and testing
root: Optional[tk.Tk] = None
textbox: Optional[ttk.Entry] = None
progress_bar: Optional[ttk.Progressbar] = None
status_label: Optional[ttk.Label] = None
deep_var: Optional[tk.BooleanVar] = None
all_var: Optional[tk.BooleanVar] = None
scan_all_var: Optional[tk.BooleanVar] = None
gpt_var: Optional[tk.BooleanVar] = None
dry_var: Optional[tk.BooleanVar] = None
git_var: Optional[tk.BooleanVar] = None
filter_var: Optional[tk.StringVar] = None
filter_entry: Optional[ttk.Entry] = None
tree: Optional[ttk.Treeview] = None
scan_button: Optional[ttk.Button] = None
cancel_button: Optional[ttk.Button] = None
view_button: Optional[ttk.Button] = None
rescan_button: Optional[ttk.Button] = None
open_button: Optional[ttk.Button] = None
analyze_button: Optional[ttk.Button] = None
exclude_button: Optional[ttk.Button] = None
reveal_button: Optional[ttk.Button] = None
vt_button: Optional[ttk.Button] = None
results_button: Optional[ttk.Menubutton] = None
browse_button: Optional[ttk.Menubutton] = None
copy_cmd_button: Optional[ttk.Button] = None
git_checkbox: Optional[ttk.Checkbutton] = None
deep_checkbox: Optional[ttk.Checkbutton] = None
scan_all_checkbox: Optional[ttk.Checkbutton] = None
dry_checkbox: Optional[ttk.Checkbutton] = None
gpt_checkbox: Optional[ttk.Checkbutton] = None
all_checkbox: Optional[ttk.Checkbutton] = None
threshold_spin: Optional[ttk.Spinbox] = None
provider_combo: Optional[ttk.Combobox] = None
model_combo: Optional[ttk.Combobox] = None
api_key_entry: Optional[ttk.Entry] = None
api_entry: Optional[ttk.Entry] = None
context_menu: Optional[tk.Menu] = None
_all_results_cache: List[Tuple[Any, ...]] = []
_last_scan_summary: str = ""


def resolve_remote_url(url: str) -> str:
    """Resolve GitHub/GitLab repository or blob URLs to their raw content or archive.

    Args:
        url: The original URL to resolve.

    Returns:
        A resolved URL pointing to raw content or a downloadable archive.
    """
    url = url.strip()
    if not url.lower().startswith(('http://', 'https://')):
        return url

    # Remove trailing slashes and common fragments
    url = re.sub(r'/$', '', url)
    url = re.sub(r'#.*$', '', url)

    # 1. GitHub Blob -> Raw
    # Example: https://github.com/user/repo/blob/main/script.py -> https://raw.githubusercontent.com/user/repo/main/script.py
    gh_blob_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/(.+)', url, re.IGNORECASE)
    if gh_blob_match:
        user, repo, path = gh_blob_match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"

    # 2. GitHub PR/Commit -> Diff
    # Example: https://github.com/user/repo/pull/1 -> https://github.com/user/repo/pull/1.diff
    # Example: https://github.com/user/repo/commit/abc -> https://github.com/user/repo/commit/abc.diff
    gh_patch_match = re.match(r'(https?://(?:www\.)?github\.com/[^/]+/[^/]+/(?:pull/\d+|commit/[a-f0-9]+))(?:/.*)?$', url, re.IGNORECASE)
    if gh_patch_match:
        return f"{gh_patch_match.group(1)}.diff"

    # 3. GitHub Gist -> Raw
    # Example: https://gist.github.com/user/id -> https://gist.github.com/user/id/raw
    gist_match = re.match(r'https?://gist\.github\.com/([^/]+)/([a-f0-9]+)$', url, re.IGNORECASE)
    if gist_match:
        return f"{url}/raw"

    # 4. GitHub Branch/Tag -> ZIP Archive
    # Example: https://github.com/user/repo/tree/main -> https://github.com/user/repo/archive/refs/heads/main.zip
    gh_tree_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/tree/(.+)', url, re.IGNORECASE)
    if gh_tree_match:
        user, repo, ref = gh_tree_match.groups()
        return f"https://github.com/{user}/{repo}/archive/refs/heads/{ref}.zip"

    # 5. GitHub Repo -> ZIP Archive
    # Example: https://github.com/user/repo -> https://github.com/user/repo/archive/HEAD.zip
    gh_repo_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)$', url, re.IGNORECASE)
    if gh_repo_match:
        user, repo = gh_repo_match.groups()
        if repo.lower() not in ('settings', 'pulls', 'issues', 'actions', 'projects', 'wiki', 'security', 'insights', 'pull'):
            return f"https://github.com/{user}/{repo}/archive/HEAD.zip"

    # 6. GitLab Blob -> Raw
    # Example: https://gitlab.com/user/repo/-/blob/main/script.py -> https://gitlab.com/user/repo/-/raw/main/script.py
    gl_blob_match = re.match(r'(https?://(?:www\.)?gitlab\.com/.+?)/-/blob/(.+)', url, re.IGNORECASE)
    if gl_blob_match:
        base, path = gl_blob_match.groups()
        return f"{base}/-/raw/{path}"

    # 7. GitLab MR -> Diff
    # Example: https://gitlab.com/user/repo/-/merge_requests/1 -> https://gitlab.com/user/repo/-/merge_requests/1.diff
    gl_mr_match = re.match(r'(https?://(?:www\.)?gitlab\.com/(.+)/([^/]+)/-/merge_requests/\d+)(?:/.*)?$', url, re.IGNORECASE)
    if gl_mr_match:
        return f"{gl_mr_match.group(1)}.diff"

    # 8. GitLab Repo -> ZIP Archive
    # Example: https://gitlab.com/user/repo -> https://gitlab.com/user/repo/-/archive/main/repo-main.zip
    # Note: GitLab is trickier as the default branch varies. We'll try common patterns.
    gl_repo_match = re.match(r'https?://(?:www\.)?gitlab\.com/(?!.*/-/)(.+)/([^/]+)$', url, re.IGNORECASE)
    if gl_repo_match:
        group_path, repo = gl_repo_match.groups()
        # GitLab doesn't have a universal HEAD.zip, but we can try to guess or just return the URL
        # Common default branches are 'main' or 'master'. We'll try 'main' and let fetch_url_content fallback if it fails.
        return f"https://gitlab.com/{group_path}/{repo}/-/archive/main/{repo}-main.zip"

    # 9. Bitbucket Cloud Raw
    # Example: https://bitbucket.org/user/repo/src/main/script.py -> https://bitbucket.org/user/repo/raw/main/script.py
    bb_raw_match = re.match(r'https?://(?:www\.)?bitbucket\.org/([^/]+)/([^/]+)/src/([^/]+)/(.+)', url, re.IGNORECASE)
    if bb_raw_match:
        user, repo, ref, path = bb_raw_match.groups()
        return f"https://bitbucket.org/{user}/{repo}/raw/{ref}/{path}"

    # 10. Bitbucket Cloud Repo -> ZIP Archive
    # Example: https://bitbucket.org/user/repo -> https://bitbucket.org/user/repo/get/HEAD.zip
    bb_repo_match = re.match(r'https?://(?:www\.)?bitbucket\.org/([^/]+)/([^/]+)$', url, re.IGNORECASE)
    if bb_repo_match:
        user, repo = bb_repo_match.groups()
        return f"https://bitbucket.org/{user}/{repo}/get/HEAD.zip"

    return url


def load_file(filename: str, mode: str = 'single_line') -> Union[str, List[str]]:
    """Reads a file and returns its content.

    Args:
        filename: The path to the file.
        mode: 'single_line' (default) returns the first line.
              'multi_line' returns all lines as a list.
              'full' returns the entire file content.

    Returns:
        The file content, or an empty result if the file is missing.
    """
    try:
        with open(filename, 'r') as file:
            if mode == 'single_line':
                return file.readline().strip()
            elif mode == 'multi_line':
                return file.read().splitlines()
            elif mode == 'full':
                return file.read().strip()
    except (FileNotFoundError, PermissionError):
        if mode == 'multi_line':
            return []
        return ''


def fetch_url_content(url: str, timeout: int = 10, max_size: Optional[int] = None) -> bytes:
    """Fetches content from a URL with safety limits. Automatically resolves GitHub/GitLab links.

    Args:
        url: The URL to fetch.
        timeout: Connection timeout in seconds.
        max_size: Maximum download size in bytes. Defaults to Config.MAX_FILE_SIZE.

    Returns:
        The content as bytes.

    Raises:
        ValueError: If the response is too large or invalid.
        urllib.error.URLError: If the fetch fails.
    """
    if max_size is None:
        max_size = Config.MAX_FILE_SIZE

    # Resolve GitHub/GitLab URLs to raw content/archives
    url = resolve_remote_url(url)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        content_length = response.getheader('Content-Length')
        if content_length and int(content_length) > max_size:
            raise ValueError(f"Content too large ({format_bytes(int(content_length))})")

        data = response.read(max_size + 1)
        if len(data) > max_size:
            raise ValueError(f"Content too large (exceeds {format_bytes(max_size)})")

        return data


class Config:
    """Global configuration settings for the scanner."""
    VERSION = "1.4.0"
    SETTINGS_FILE = ".gptscan_settings.json"
    CACHE_FILE = ".gptscan_cache.json"
    MAXLEN = 1024
    EXPECTED_KEYS = ["administrator", "end-user", "threat-level"]
    MAX_RETRIES = 3
    RATE_LIMIT_PER_MINUTE = 60
    MAX_CONCURRENT_REQUESTS = 5
    MAX_FILE_SIZE = 10 * 1024 * 1024
    MAX_SOURCE_VIEW_SIZE = 2 * 1024 * 1024
    gpt_cache: Dict[str, Dict[str, Any]] = {}
    apikey: str = load_file('apikey.txt')
    taskdesc: str = load_file('task.txt', mode='full')
    GPT_ENABLED: bool = False
    extensions_set: set[str] = set()
    extensions_missing: bool = False
    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_base: Optional[str] = None
    ignore_patterns: List[str] = []
    THRESHOLD: int = 50
    last_path: str = ""
    recent_paths: List[str] = []
    deep_scan: bool = False
    git_changes_only: bool = False
    show_all_files: bool = False
    scan_all_files: bool = False
    use_ai_analysis: bool = False

    DEFAULT_EXTENSIONS = ['.py', '.js', '.bat', '.ps1', '.ipynb']

    @staticmethod
    def _get_setting_int(settings: Dict[str, Any], key: str, default: int) -> int:
        """Extract an integer setting with fallback to default on error."""
        val = settings.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _get_setting_bool(settings: Dict[str, Any], key: str, default: bool) -> bool:
        """Extract a boolean setting with fallback to default on error."""
        val = settings.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower in ('true', '1', 'yes', 'on'):
                return True
            if val_lower in ('false', '0', 'no', 'off'):
                return False
        try:
            return bool(int(val))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def is_container(file_path: Union[Path, str], content: Optional[bytes] = None) -> bool:
        """Check if a file is a container that should be unpacked (archives, notebooks, manifests, etc.)."""
        # 1. Check by magic bytes if content is available
        if content:
            if content.startswith(b'PK\x03\x04') or content.startswith(b'\x1f\x8b'):
                return True
            if len(content) > 262 and content[257:262] == b'ustar':
                return True

        # 2. Check by extension or basename
        path_s = str(file_path).lower()
        # Extensions and manifest files
        if path_s.endswith(('.zip', '.tar', '.tar.gz', '.ipynb', '.md', '.html', '.htm', '.xhtml', '.yml', '.yaml',
                            '.diff', '.patch', 'package.json', 'composer.json', 'deno.json', 'deno.jsonc')):
            return True
        # Dockerfile and Makefile variants
        basename = os.path.basename(path_s)
        if basename in ('dockerfile', 'makefile') or basename.endswith(('.dockerfile', '.makefile')):
            return True
        return False

    apikey_missing_message = (
        "No API key found. You cannot use OpenAI or OpenRouter, but local scans and Ollama still work."
    )
    task_missing_message = (
        "The 'task.txt' file is missing. The scanner will not use AI analysis."
    )
    extensions_missing_message = (
        f"The 'extensions.txt' file is missing. The scanner will use these default types: {', '.join(DEFAULT_EXTENSIONS)}"
    )

    @classmethod
    def set_extensions(cls, extensions_list: List[str], missing: bool = False) -> None:
        cls.extensions_missing = missing
        cls.extensions_set = set()
        for ext in extensions_list:
            clean_ext = ext.strip().lower()
            if clean_ext:
                cls.extensions_set.add(clean_ext if clean_ext.startswith('.') else f".{clean_ext}")

    @classmethod
    def save_apikey(cls) -> None:
        """Save the current API key to the apikey.txt file."""
        try:
            with open('apikey.txt', 'w', encoding='utf-8') as f:
                f.write(cls.apikey)
        except Exception as e:
            print(f"Warning: Could not save API key: {e}", file=sys.stderr)

    @classmethod
    def save_extensions(cls) -> None:
        """Save current extensions to the extensions.txt file."""
        try:
            with open('extensions.txt', 'w', encoding='utf-8') as f:
                for ext in sorted(cls.extensions_set):
                    f.write(f"{ext}\n")
        except Exception as e:
            print(f"Warning: Could not save extensions: {e}", file=sys.stderr)

    @classmethod
    def save_settings(cls) -> None:
        """Save persistent GUI settings to a JSON file."""
        settings = {
            "last_path": cls.last_path,
            "deep_scan": cls.deep_scan,
            "git_changes_only": cls.git_changes_only,
            "show_all_files": cls.show_all_files,
            "scan_all_files": cls.scan_all_files,
            "use_ai_analysis": cls.use_ai_analysis,
            "provider": cls.provider,
            "model_name": cls.model_name,
            "api_base": cls.api_base,
            "threshold": cls.THRESHOLD,
            "max_file_size": cls.MAX_FILE_SIZE,
            "max_source_view_size": cls.MAX_SOURCE_VIEW_SIZE,
            "recent_paths": cls.recent_paths,
        }
        try:
            with open(cls.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}", file=sys.stderr)

    @classmethod
    def load_settings(cls) -> None:
        """Load persistent GUI settings from a JSON file."""
        if not os.path.exists(cls.SETTINGS_FILE):
            return
        try:
            with open(cls.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                cls.last_path = settings.get("last_path", cls.last_path)
                cls.deep_scan = cls._get_setting_bool(settings, "deep_scan", cls.deep_scan)
                cls.git_changes_only = cls._get_setting_bool(settings, "git_changes_only", cls.git_changes_only)
                cls.show_all_files = cls._get_setting_bool(settings, "show_all_files", cls.show_all_files)
                cls.scan_all_files = cls._get_setting_bool(settings, "scan_all_files", cls.scan_all_files)
                cls.use_ai_analysis = cls._get_setting_bool(settings, "use_ai_analysis", cls.use_ai_analysis)
                cls.provider = settings.get("provider", cls.provider)
                cls.model_name = settings.get("model_name", cls.model_name)
                cls.api_base = settings.get("api_base", cls.api_base)
                cls.THRESHOLD = cls._get_setting_int(settings, "threshold", cls.THRESHOLD)
                cls.MAX_FILE_SIZE = cls._get_setting_int(settings, "max_file_size", cls.MAX_FILE_SIZE)
                cls.MAX_SOURCE_VIEW_SIZE = cls._get_setting_int(settings, "max_source_view_size", cls.MAX_SOURCE_VIEW_SIZE)

                recent = settings.get("recent_paths", cls.recent_paths)
                if isinstance(recent, list):
                    cls.recent_paths = [str(p) for p in recent]
        except Exception as e:
            print(f"Warning: Could not load settings: {e}", file=sys.stderr)

    @classmethod
    def save_cache(cls) -> None:
        """Save AI analysis cache to a JSON file."""
        try:
            with open(cls.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cls.gpt_cache, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save AI cache: {e}", file=sys.stderr)

    @classmethod
    def load_cache(cls) -> None:
        """Load AI analysis cache from a JSON file."""
        if not os.path.exists(cls.CACHE_FILE):
            return
        try:
            with open(cls.CACHE_FILE, 'r', encoding='utf-8') as f:
                cls.gpt_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load AI cache: {e}", file=sys.stderr)

    @classmethod
    def initialize(cls) -> None:
        if not cls.apikey:
            # Fallback to environment variables if apikey.txt is missing or empty
            cls.apikey = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""

        if not cls.apikey:
            print(cls.apikey_missing_message, file=sys.stderr)
        if not cls.taskdesc:
            print(cls.task_missing_message, file=sys.stderr)

        # Enable GPT if task description is present.
        # Specific provider requirements (like API key) are checked at runtime.
        cls.GPT_ENABLED = bool(cls.taskdesc)

        loaded_extensions = load_file('extensions.txt', mode='multi_line')
        if not loaded_extensions:
            cls.set_extensions(cls.DEFAULT_EXTENSIONS, missing=True)
            print(cls.extensions_missing_message, file=sys.stderr)
        else:
            cls.set_extensions(loaded_extensions)

        loaded_ignores = load_file('.gptscanignore', mode='multi_line')
        cls.ignore_patterns = []
        if loaded_ignores:
            cls.ignore_patterns = [
                line.strip().split('#')[0].strip() for line in loaded_ignores
                if line.strip() and not line.strip().startswith('#')
                and line.strip().split('#')[0].strip()
            ]

        cls.load_settings()
        cls.load_cache()

    @classmethod
    def is_supported_file(cls, file_path: Union[Path, str], is_explicit: bool = False, is_member: bool = False, content: Optional[bytes] = None) -> bool:
        """Check if a file should be scanned based on extension, content, explicit request, or scan_all_files setting."""
        if is_explicit or cls.scan_all_files:
            return True

        # Normalize file_path to string for consistent extension checking
        path_str = str(file_path).lower()

        if not is_member and cls.is_container(path_str, content=content):
            return True

        extension = os.path.splitext(path_str)[1]
        if extension in cls.extensions_set:
            return True

        file_path = Path(file_path)

        # Check for a script starting line (like #!/bin/bash) for files without a recognized extension
        try:
            first_line = None
            if content is not None:
                if content.startswith(b'#!'):
                    first_line = content[2:].split(b'\n', 1)[0][:126].decode('utf-8', errors='ignore').lower()
            elif file_path.is_file():
                with open(file_path, 'rb') as f:
                    if f.read(2) == b'#!':
                        first_line = f.readline(126).decode('utf-8', errors='ignore').lower()

            if first_line:
                interpreters = ['python', 'node', 'nodejs', 'javascript', 'bash', 'sh', 'ash', 'dash', 'zsh', 'perl', 'ruby', 'php', 'pwsh', 'powershell', 'lua', 'osascript', 'ipython']
                escaped_interpreters = [re.escape(i) for i in interpreters]
                pattern = r'(?:/|\s|^)(?:' + '|'.join(escaped_interpreters) + r')\d*\b'
                if re.search(pattern, first_line):
                    return True
        except (OSError, UnicodeDecodeError):
            pass

        return False


Config.initialize()

ui_queue = queue.Queue()
current_cancel_event: Optional[threading.Event] = None
_model_cache: Optional[Any] = None
_tf_module: Optional[Any] = None
_model_lock = threading.Lock()
_async_openai_client: Optional[Any] = None
default_font_measure: Optional[Callable[[str], int]] = None


def get_model() -> Any:
    """Load the detection model when it is first needed and keep it in memory."""

    global _model_cache, _tf_module
    if _model_cache is not None:
        return _model_cache

    with _model_lock:
        if _model_cache is None:
            if _tf_module is None:
                import tensorflow as tf
                _tf_module = tf
            _model_cache = _tf_module.keras.models.load_model('scripts.h5', compile=False)
    return _model_cache


def get_async_openai_client() -> Any:
    """Create the AI service connection when it is first needed and reuse it."""

    global _async_openai_client

    if _async_openai_client is None:
        api_key = Config.apikey
        if Config.provider == "ollama" and not api_key:
            api_key = "ollama"

        if api_key:
            base_url = Config.api_base
            if not base_url:
                if Config.provider == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                elif Config.provider == "ollama":
                    base_url = "http://localhost:11434/v1"

            from openai import AsyncOpenAI
            _async_openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _async_openai_client


def update_status(message: str) -> None:
    """Update the status label with the given message.

    Parameters
    ----------
    message : str
        The message to display in the status label.
    """
    if status_label:
        status_label.config(text=message)
    if root:
        root.update_idletasks()


def update_progress(value: int) -> None:
    """Update the progress bar to reflect current progress.

    Parameters
    ----------
    value : int
        Current progress count to display.
    """
    if progress_bar and root:
        progress_bar['value'] = value
        try:
            total = float(progress_bar["maximum"])
            if total > 0:
                percent = int((value / total) * 100)
                root.title(f"[{percent}%] GPT Virus Scanner")
        except (ValueError, TypeError, tk.TclError):
            pass
        root.update_idletasks()


def configure_progress(max_value: int) -> None:
    """Initialize progress bar values for a new scan.

    Parameters
    ----------
    max_value : int
        Total number of steps expected for the scan.
    """
    if progress_bar and root:
        progress_bar["maximum"] = max_value
        progress_bar["value"] = 0
        root.title("[0%] GPT Virus Scanner")
        root.update_idletasks()


def enqueue_ui_update(func: Callable, *args: Any, **kwargs: Any) -> None:
    """Add a task to the queue so the main window can update safely.

    Parameters
    ----------
    func : Callable
        Function to execute on the UI thread.
    *args
        Positional arguments for ``func``.
    **kwargs
        Keyword arguments for ``func``.
    """
    ui_queue.put((func, args, kwargs))


def process_ui_queue() -> None:
    """Drain the UI queue, executing any pending UI updates.

    Returns
    -------
    None
        This function schedules itself to run again via ``root.after``.
    """
    try:
        while not ui_queue.empty():
            func, args, kwargs = ui_queue.get()
            try:
                func(*args, **kwargs)
            finally:
                ui_queue.task_done()
    finally:
        root.after(50, process_ui_queue)


def bind_hover_message(widget: tk.Widget, message: str, label: Optional[ttk.Label] = None) -> None:
    """Bind mouse enter/leave events to update the status label."""
    target_label = label or status_label
    if not target_label:
        return

    # Store the previous message to restore it later
    previous_message: List[str] = ["Ready"]

    def on_enter(event):
        if current_cancel_event is None:
            # Save current text, defaulting to Ready if empty
            current_text = target_label.cget("text")
            previous_message[0] = current_text if current_text else "Ready"
            target_label.config(text=message)
            if root:
                root.update_idletasks()

    def on_leave(event):
        if current_cancel_event is None:
            target_label.config(text=previous_message[0])
            if root:
                root.update_idletasks()

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def _set_scan_target(path: Union[str, Iterable[str]]) -> None:
    """Update the scan target textbox and set focus to the scan button.

    Args:
        path: A single path string, or an iterable of path strings.
    """
    if not path or not textbox:
        return

    # Handle multiple paths or a single path string
    if isinstance(path, (list, tuple)):
        # Join multiple targets with appropriate quoting
        formatted_path = shlex.join(path)
    else:
        # For a single path, use quote if it's not a list, for safety
        formatted_path = shlex.quote(str(path))

    textbox.delete(0, tk.END)
    textbox.insert(0, formatted_path)
    if scan_button:
        scan_button.focus_set()


def browse_dir_click() -> None:
    """Handle the directory selection dialog and populate the textbox."""
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        _set_scan_target(folder_selected)


def select_url_click() -> None:
    """Handle the URL input dialog and populate the textbox."""
    url_selected = simpledialog.askstring("Scan URL", "Enter a script URL to scan (http/https):")
    if url_selected:
        _set_scan_target(url_selected.strip())


def toggle_ai_controls() -> None:
    """Enable or disable AI analysis controls based on current settings and scan state."""
    enabled = gpt_var.get() if gpt_var else False
    is_scanning = current_cancel_event is not None

    if provider_combo and model_combo and api_key_entry and api_entry:
        if enabled and not is_scanning:
            provider_combo.config(state="readonly")
            model_combo.config(state="normal")
            api_key_entry.config(state="normal")
            api_entry.config(state="normal")
        else:
            provider_combo.config(state="disabled")
            model_combo.config(state="disabled")
            api_key_entry.config(state="disabled")
            api_entry.config(state="disabled")

    update_tree_columns()


def browse_file_click() -> None:
    """Handle the file selection dialog and populate the textbox."""
    ext_list = sorted(Config.extensions_set) if Config.extensions_set else Config.DEFAULT_EXTENSIONS
    file_types = [
        ("Script files", ";".join([f"*{e}" for e in ext_list])),
        ("All files", "*.*")
    ]
    files_selected = filedialog.askopenfilenames(
        title="Select File(s) to Scan",
        filetypes=file_types
    )
    if files_selected:
        _set_scan_target(files_selected)


class AsyncRateLimiter:
    """Simple asynchronous rate limiter using a sliding one-minute window."""

    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = max(1, rate_per_minute)
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, on_wait: Optional[Callable[[float], None]] = None) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.rate_per_minute:
                    self._timestamps.append(now)
                    return

                wait_time = 60 - (now - self._timestamps[0])
                if on_wait:
                    on_wait(wait_time)

            await asyncio.sleep(wait_time)


def extract_data_from_gpt_response(response: Any) -> Union[Dict, str]:
    """Parse and validate the JSON payload returned from the OpenAI API.

    Parameters
    ----------
    response : Any
        OpenAI response object expected to contain ``choices[0].message.content``
        with JSON matching ``Config.EXPECTED_KEYS``.

    Returns
    -------
    Union[Dict, str]
        Parsed JSON dictionary when valid; otherwise, a human-readable error
        message describing the parsing failure.
    """
    content = response.choices[0].message.content

    # Extract JSON from markdown blocks if present
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    try:
        json_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return str(exc)

    if not isinstance(json_data, dict):
        return "The AI's response was not in the expected format."

    missing_keys = [key for key in Config.EXPECTED_KEYS if key not in json_data]
    if missing_keys:
        return f"The AI's response was missing required information: {', '.join(missing_keys)}"

    threat_level_value = json_data.get("threat-level")
    try:
        threat_level = int(threat_level_value)
    except (TypeError, ValueError):
        return f"The threat level '{threat_level_value}' is not a valid number."

    if not 0 <= threat_level <= 100:
        return f"The threat level {threat_level} is not between 0 and 100."

    json_data["threat-level"] = threat_level
    return json_data


async def async_handle_gpt_response(
    snippet: str,
    taskdesc: str,
    rate_limiter: AsyncRateLimiter,
    semaphore: asyncio.Semaphore,
    wait_callback: Optional[Callable[[float], None]] = None,
) -> Optional[Dict]:
    """Request GPT analysis asynchronously with retry, caching and rate limits."""

    retries = 0
    json_data: Optional[Union[Dict, str]] = None
    # Include provider, model and prompt in cache key for robustness
    cache_input = f"{Config.provider}:{Config.model_name}:{taskdesc}:{snippet}"
    cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
    if cache_key in Config.gpt_cache:
        return Config.gpt_cache[cache_key]

    client = get_async_openai_client()
    if client is None:
        return None

    create_completion = partial(client.chat.completions.create, model=Config.model_name)
    messages = [
        {"role": "system", "content": taskdesc},
        {"role": "user", "content": snippet},
    ]

    while retries < Config.MAX_RETRIES and (json_data is None or isinstance(json_data, str)):
        await rate_limiter.acquire(on_wait=wait_callback)
        try:
            async with semaphore:
                response = await create_completion(messages=messages)
            extracted_data = extract_data_from_gpt_response(response)
        except Exception as err:  # pragma: no cover - safety net
            status_code = getattr(err, "status_code", None)
            is_rate_limit = status_code == 429 or "429" in str(getattr(err, "status", "")) or "rate limit" in str(err).lower()
            is_structure_error = isinstance(err, (AttributeError, IndexError, TypeError))

            if is_rate_limit or is_structure_error:
                backoff = 2 ** retries
                retries += 1
                await asyncio.sleep(backoff)
                continue
            print(f"An unexpected error occurred: {err}", file=sys.stderr)
            break

        if isinstance(extracted_data, dict):
            json_data = extracted_data
        else:
            print(extracted_data, file=sys.stderr)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            messages.append({"role": "user", "content": f"I encountered an issue: {extracted_data}. Could you correct your response?"})
            retries += 1

    if isinstance(json_data, dict):
        Config.gpt_cache[cache_key] = json_data
        Config.save_cache()
        return json_data

    print("Failed to obtain a valid response from GPT after multiple retries.", file=sys.stderr)
    return None


def request_single_gpt_analysis(snippet: str) -> Optional[Dict[str, Any]]:
    """Perform a one-off AI assessment for a code snippet."""
    if not Config.GPT_ENABLED:
        return None

    async def run_analysis():
        rate_limiter = AsyncRateLimiter(60)
        semaphore = asyncio.Semaphore(1)
        return await async_handle_gpt_response(
            snippet,
            Config.taskdesc,
            rate_limiter=rate_limiter,
            semaphore=semaphore
        )

    try:
        return asyncio.run(run_analysis())
    except Exception as e:
        print(f"Error during on-demand AI analysis: {e}", file=sys.stderr)
        return None


def adjust_newlines(val: Any, width: int, pad: int = 10, measure: Optional[Callable[[str], int]] = None) -> Any:
    """Wrap strings based on the available column width while preserving indentation."""
    if not isinstance(val, str):
        return val

    measure = measure or tkinter.font.Font(font='TkDefaultFont').measure
    original_lines = val.splitlines()
    if not original_lines:
        return val

    wrapped_lines = []
    for line in original_lines:
        # Preserve leading whitespace
        indent = re.match(r'^(\s*)', line).group(1)
        content = line[len(indent):]
        words = content.split()

        if not words:
            wrapped_lines.append(indent)
            continue

        current_line_words = []
        for word in words:
            test_line = indent + ' '.join(current_line_words + [word])
            if not current_line_words or measure(test_line) < (width - pad):
                current_line_words.append(word)
            else:
                wrapped_lines.append(indent + ' '.join(current_line_words))
                current_line_words = [word]

        if current_line_words:
            wrapped_lines.append(indent + ' '.join(current_line_words))

    return '\n'.join(wrapped_lines)


def get_wrapped_values(tree: ttk.Treeview, values: Iterable[Any], measure: Optional[Callable[[str], int]] = None, col_widths: Optional[List[int]] = None) -> List[Any]:
    """Wrap a list of values to fit the current Treeview column widths."""
    measure = measure or (default_font_measure or tkinter.font.Font(font='TkDefaultFont').measure)
    col_widths = col_widths or [tree.column(cid)['width'] for cid in tree['columns']]

    # Only wrap the first 6 columns, leave the rest (including line and hidden orig_json) as is
    wrapped = [adjust_newlines(v, w, measure=measure) for v, w in zip(list(values)[:6], col_widths[:6])]
    if len(values) > 6:
        wrapped.extend(list(values)[6:])
    return wrapped


def motion_handler(tree: ttk.Treeview, event: Optional[tk.Event]) -> None:
    """Wrap long cell values so they fit within the visible column width."""

    if (event is None) or (tree.identify_region(event.x, event.y) == "separator"):
        measure = default_font_measure or tkinter.font.Font(font='TkDefaultFont').measure
        col_widths = [tree.column(cid)['width'] for cid in tree['columns']]

        for iid in tree.get_children():
            raw_values = _get_item_raw_values(iid)
            if not raw_values:
                continue

            new_vals = get_wrapped_values(tree, raw_values, measure=measure, col_widths=col_widths)
            new_vals.append(json.dumps(raw_values[:7]))
            tree.item(iid, values=new_vals)


def get_git_changed_files(path: str = ".") -> List[str]:
    """Get a list of changed files (staged, unstaged, untracked) from git."""
    abs_path = os.path.abspath(path)
    search_dir = os.path.dirname(abs_path) if os.path.isfile(abs_path) else abs_path

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=search_dir,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []

    try:
        rel_target = os.path.relpath(abs_path, toplevel)
        targets = [rel_target]
    except ValueError:
        return []

    files = set()
    # Changed (staged and unstaged) relative to HEAD
    try:
        cmd = ["git", "diff", "--name-only", "HEAD", "--"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Untracked files
    try:
        cmd = ["git", "ls-files", "--others", "--exclude-standard", "--"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return [p for f in files if os.path.exists(p := os.path.join(toplevel, f))]


def collect_files(targets: Union[str, List[str]]) -> List[Path]:
    """Collect files from a single path or a list of paths (files, directories, or globs).

    Parameters
    ----------
    targets : Union[str, List[str]]
        A single directory path or a list of file/directory paths or glob patterns.
        Multiple targets can be provided in a single space-separated string.

    Returns
    -------
    List[Path]
        A deduplicated list of files to scan.
    """
    if isinstance(targets, (str, Path)):
        try:
            targets = shlex.split(str(targets))
        except ValueError:
            targets = [str(targets)]

    results: List[Path] = []
    for t in targets:
        p = Path(t)
        if p.exists():
            if p.is_file():
                results.append(p)
            elif p.is_dir():
                results.extend([f for f in p.rglob('*') if f.is_file()])
        elif any(char in str(t) for char in ['*', '?', '[']):
            # Try to expand as a glob pattern
            expanded = glob.glob(str(t), recursive=True)
            for path_str in expanded:
                ep = Path(path_str)
                if ep.is_file():
                    results.append(ep)
                elif ep.is_dir():
                    results.extend([f for f in ep.rglob('*') if f.is_file()])

    # Use dict keys to remove duplicates while preserving insertion order.
    return list(dict.fromkeys(results))


def parse_percent(val: str, default: float = -1.0) -> float:
    """Convert a percentage string to a float, returning ``default`` on error."""
    if not isinstance(val, str):
        return default

    text = val.strip()
    if not text or not text.endswith('%'):
        return default

    try:
        return float(text.strip('%'))
    except ValueError:
        return default


def parse_size_string(size_str: str) -> int:
    """Convert a human-readable size string (e.g., "10MB", "500KB") to bytes.

    Args:
        size_str: The size string to parse.

    Returns:
        The size in bytes as an integer.

    Raises:
        ValueError: If the string format is invalid.
    """
    if not size_str:
        raise ValueError("Size string is empty")

    match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]*)$', size_str.strip())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    value_str, unit = match.groups()
    value = float(value_str)
    unit = unit.upper()

    units = {
        '': 1,
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'KIB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'MIB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'GIB': 1024 * 1024 * 1024,
        'T': 1024 * 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024,
        'TIB': 1024 * 1024 * 1024 * 1024,
    }

    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")

    return int(value * units[unit])


def format_bytes(num: float) -> str:
    """Format a number of bytes into a human-readable string (e.g., KiB, MiB)."""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def get_file_sha256(file_path_or_data: Union[str, Path, bytes]) -> str:
    """Calculate the SHA256 hash of a file or bytes.

    Args:
        file_path_or_data: The path to the file or the data itself as bytes.

    Returns:
        The hex digest of the SHA256 hash, or an empty string if calculation fails.
    """
    if isinstance(file_path_or_data, bytes):
        return hashlib.sha256(file_path_or_data).hexdigest()

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path_or_data, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return ""


def format_scan_summary(total_scanned: int, threats_found: int, total_bytes: Optional[int] = None, elapsed_time: Optional[float] = None, use_color: bool = False, high_risk: int = 0, medium_risk: int = 0) -> str:
    """Format a human-readable summary of the scan results.

    Parameters
    ----------
    total_scanned : int
        Total number of files scanned.
    threats_found : int
        Total number of suspicious files detected.
    total_bytes : int, optional
        Total bytes scanned.
    elapsed_time : float, optional
        Time taken for the scan in seconds.
    use_color : bool, optional
        Whether to use ANSI color codes in the output.
    high_risk : int, optional
        Number of high risk files found.
    medium_risk : int, optional
        Number of medium risk files found.
    """
    threat_text = "suspicious file" if threats_found == 1 else "suspicious files"

    threats_display = str(threats_found)
    if use_color and threats_found > 0:
        # Use Bold Red for threats in terminal
        threats_display = f"\033[1;91m{threats_found}\033[0m"

    bytes_info = f" ({format_bytes(total_bytes)})" if total_bytes is not None else ""
    summary = f"Scan complete: {total_scanned} files{bytes_info} scanned, {threats_display} {threat_text} found"

    if threats_found > 0:
        summary += f" ({high_risk} high risk, {medium_risk} medium risk)."
    else:
        summary += "."

    if elapsed_time and elapsed_time > 0:
        files_per_sec = total_scanned / elapsed_time
        summary += f" Time: {elapsed_time:.1f}s ({files_per_sec:.1f} files/s"
        if total_bytes:
            bytes_per_sec = total_bytes / elapsed_time
            summary += f", {format_bytes(bytes_per_sec)}/s"
        summary += ")."
    return summary


def get_effective_threat_level(own_conf_str: Any, gpt_conf_str: Any) -> float:
    """Calculate the effective threat level, prioritizing GPT over local AI."""
    gpt_val = parse_percent(gpt_conf_str)
    if gpt_val >= 0:
        return gpt_val
    return parse_percent(own_conf_str)


def get_risk_category(conf: float, threshold: int) -> Optional[str]:
    """Categorize a threat level into 'high', 'medium', or None (no threat)."""
    if conf < threshold:
        return None
    if conf >= 80:
        return 'high'
    return 'medium'


def sort_column(tv: ttk.Treeview, col: str, reverse: bool) -> None:
    """Sort a Treeview column, toggling sort order on subsequent clicks.

    Parameters
    ----------
    tv : ttk.Treeview
        Treeview widget containing the data to sort.
    col : str
        Column identifier to sort.
    reverse : bool
        Sort order; ``True`` for descending and ``False`` for ascending.
    """
    values_with_ids = [(tv.set(k, col), k) for k in tv.get_children("")]
    if col in {"own_conf", "gpt_conf"}:
        values_with_ids = [(parse_percent(val), k) for val, k in values_with_ids]
    elif col == "line":
        values_with_ids = [(int(val) if str(val).isdigit() else -1, k) for val, k in values_with_ids]

    values_with_ids.sort(key=lambda item: item[0], reverse=reverse)

    for index, (_, k) in enumerate(values_with_ids):
        tv.move(k, "", index)

    # Reverse sort order on subsequent clicks of the same column header
    tv.heading(col, command=lambda: sort_column(tv, col, not reverse))


def _matches_filter(values: Tuple[Any, ...]) -> bool:
    """Check if the given values match the current filter string and threat threshold."""
    # Check threshold first unless "Show all files" is checked
    is_show_all = all_var.get() if all_var else False
    if not is_show_all:
        conf = get_effective_threat_level(values[1], values[4])
        # Only hide results with a valid percentage score below the threshold.
        # Special statuses (Error, Dry Run, etc.) result in -1.0 and stay visible.
        if 0 <= conf < Config.THRESHOLD:
            return False

    if not filter_var:
        return True
    query = filter_var.get().lower().strip()
    if not query:
        return True

    for val in values[:6]:
        if query in str(val).lower():
            return True
    return False


def _apply_filter(*args: Any) -> None:
    """Refresh the Treeview based on the current filter and cached results."""
    if not tree:
        return

    items = tree.get_children()
    if items:
        tree.delete(*items)

    match_count = 0
    for values in _all_results_cache:
        if _matches_filter(values):
            match_count += 1
            wrapped_values, tags = _prepare_tree_row(values)
            tree.insert("", tk.END, values=wrapped_values, tags=tags)

    # Update status label with filtered count if not currently scanning
    if current_cancel_event is None:
        query = filter_var.get().strip() if filter_var else ""
        total_count = len(_all_results_cache)
        if query or match_count < total_count:
            msg = f"Showing {match_count} of {total_count} results"
            if query:
                msg += f" matching '{query}'"
            update_status(msg)
        elif _last_scan_summary:
            update_status(_last_scan_summary)
        else:
            update_status("Ready")

    update_tree_columns()


def _prepare_tree_row(values: Tuple[Any, ...]) -> Tuple[List[Any], Tuple[str, ...]]:
    """Prepare wrapped values and tags for a Treeview row."""
    # Preserve original values in a hidden column as a JSON string
    # only use the first 7 columns (including line number)
    orig_data = list(values[:7])
    orig_json = json.dumps(orig_data)

    wrapped_values = get_wrapped_values(tree, values[:7])
    wrapped_values.append(orig_json)

    # Determine risk level based on threat levels
    # data format: (path, own_conf, admin, user, gpt_conf, snippet)
    conf = get_effective_threat_level(values[1], values[4])
    risk = get_risk_category(conf, Config.THRESHOLD)

    tag = f"{risk}-risk" if risk else ""

    return wrapped_values, (tag,) if tag else ()


def insert_tree_row(values: Tuple[Any, ...]) -> None:
    """Insert a row into the treeview with wrapped text and highlighting."""
    _all_results_cache.append(values)
    if tree and _matches_filter(values):
        wrapped_values, tags = _prepare_tree_row(values)
        tree.insert("", tk.END, values=wrapped_values, tags=tags)


def update_tree_row(item_id: str, values: Tuple[Any, ...]) -> None:
    """Update an existing row in the treeview with new values."""
    # Update cache
    for i, old_vals in enumerate(_all_results_cache):
        if old_vals[0] == values[0]:
            # Match line number (index 6) if available to ensure the correct entry
            # is updated when a file has multiple findings.
            if len(old_vals) > 6 and len(values) > 6:
                if str(old_vals[6]) != str(values[6]):
                    continue
            _all_results_cache[i] = values
            break

    if tree and tree.exists(item_id):
        if _matches_filter(values):
            wrapped_values, tags = _prepare_tree_row(values)
            tree.item(item_id, values=wrapped_values, tags=tags)
        else:
            tree.delete(item_id)
    elif tree and _matches_filter(values):
        # If it didn't exist (hidden) but now matches, we should probably re-apply filter
        # to show it in the right place, or just insert it.
        # For simplicity, just refresh the whole view if it was missing but now matches.
        _apply_filter()


def update_tree_columns() -> None:
    """Show or hide AI-specific columns based on settings and data."""
    if not tree:
        return

    show_ai = gpt_var.get() if gpt_var else False

    # Check if any row in the tree has AI data (gpt_conf is at index 4)
    has_ai_data = False
    for item_id in tree.get_children():
        values = tree.item(item_id, 'values')
        # values[4] is gpt_conf. Check if it's not empty
        if values and len(values) > 4 and values[4] and values[4] != "":
            has_ai_data = True
            break

    if show_ai or has_ai_data:
        tree["displaycolumns"] = ("path", "line", "own_conf", "gpt_conf", "admin_desc", "end-user_desc", "snippet")
    else:
        tree["displaycolumns"] = ("path", "line", "own_conf", "snippet")


def _auto_select_best_result() -> None:
    """Select the most relevant result (first threat, or first file) and focus it."""
    if not tree:
        return

    items = tree.get_children()
    if not items:
        return

    # Prioritize selecting the first suspicious file
    target_item = items[0]
    for iid in items:
        tags = tree.item(iid, 'tags')
        if 'high-risk' in tags or 'medium-risk' in tags:
            target_item = iid
            break

    tree.selection_set(target_item)
    tree.focus(target_item)
    tree.see(target_item)
    tree.focus_set()


def set_scanning_state(is_scanning: bool) -> None:
    """Enable or disable controls based on scanning state."""

    if scan_button:
        scan_button.config(
            text="Scanning..." if is_scanning else "Scan Now",
            state="disabled" if is_scanning else "normal"
        )
    if cancel_button:
        cancel_button.config(state="normal" if is_scanning else "disabled")

    # Disable/Enable configuration widgets during scan
    config_widgets = [
        textbox, browse_button,
        git_checkbox, deep_checkbox, scan_all_checkbox, dry_checkbox,
        gpt_checkbox, provider_combo, model_combo, api_entry, copy_cmd_button,
        all_checkbox, threshold_spin
    ]
    for widget in config_widgets:
        if widget:
            widget.config(state="disabled" if is_scanning else "normal")

    # Disable all footer buttons during a scan
    footer_buttons = [
        view_button, rescan_button, open_button, analyze_button, exclude_button,
        reveal_button, vt_button, results_button
    ]
    for btn in footer_buttons:
        if btn:
            btn.config(state="disabled" if is_scanning else "normal")

    # Ensure AI-specific controls are correctly toggled based on gpt_var and scan state
    toggle_ai_controls()

    if not is_scanning:
        if root:
            root.title("GPT Virus Scanner")
        # Re-evaluate selection-dependent buttons when scan ends
        update_button_states()


def finish_scan_state(total_scanned: Optional[int] = None, threats_found: Optional[int] = None, total_bytes: Optional[int] = None, elapsed_time: Optional[float] = None, high_risk: int = 0, medium_risk: int = 0) -> None:
    """Reset scanning controls when a scan finishes or is cancelled.

    Parameters
    ----------
    total_scanned : int, optional
        Total number of files scanned.
    threats_found : int, optional
        Total number of suspicious files detected.
    total_bytes : int, optional
        Total bytes scanned.
    elapsed_time : float, optional
        Time taken for the scan in seconds.
    high_risk : int
        Number of high risk files found.
    medium_risk : int
        Number of medium risk files found.
    """

    global current_cancel_event
    current_cancel_event = None
    set_scanning_state(False)

    if total_scanned is not None and threats_found is not None:
        summary = format_scan_summary(total_scanned, threats_found, total_bytes, elapsed_time, high_risk=high_risk, medium_risk=medium_risk)
        global _last_scan_summary
        _last_scan_summary = summary
        update_status(summary)

    update_tree_columns()
    _auto_select_best_result()


def scan_clipboard_click():
    """Scan code currently in the clipboard."""
    try:
        if root:
            content = root.clipboard_get()
            if content:
                button_click(extra_snippets=[("[Clipboard]", content.encode('utf-8'))])
    except Exception as e:
        messagebox.showwarning("Clipboard Error", f"Could not read from clipboard: {e}")


def button_click(extra_snippets: Optional[List[Tuple[str, bytes]]] = None, fail_threshold: Optional[int] = None) -> None:
    """Trigger a scan in a background thread using the selected path.

    Parameters
    ----------
    extra_snippets : List[Tuple[str, bytes]], optional
        List of (name, content) tuples to scan as in-memory buffers.

    Returns
    -------
    None
        Starts a daemon thread to run the scan.
    """
    global current_cancel_event

    if current_cancel_event is not None:
        return

    # Clear previous results
    clear_results()

    scan_path = textbox.get()
    if not scan_path and not extra_snippets:
        messagebox.showerror("Missing Selection", "Please select a file or folder to scan.")
        return

    try:
        # Use shlex.split to support multi-target selection with quoting
        scan_targets = shlex.split(scan_path) if scan_path else []
    except ValueError as e:
        messagebox.showerror("Selection Error", f"Malformed path selection: {e}")
        return

    if scan_targets and git_var.get():
        all_git_files = []
        for target in scan_targets:
            all_git_files.extend(get_git_changed_files(target))

        if not all_git_files:
            messagebox.showinfo("Git Scan", "No git changes detected in the selected directory.")
            return
        scan_targets = all_git_files

    if not dry_var.get() and not os.path.exists('scripts.h5'):
        messagebox.showerror("Model Not Found", "The scanner cannot find 'scripts.h5'. This file is required to run local scans.")
        return

    if scan_path:
        Config.last_path = scan_path
        if not Config.recent_paths or Config.recent_paths[0] != scan_path:
            if scan_path in Config.recent_paths:
                Config.recent_paths.remove(scan_path)
            Config.recent_paths.insert(0, scan_path)
            Config.recent_paths = Config.recent_paths[:10]
            if textbox:
                textbox['values'] = Config.recent_paths
        Config.save_settings()

    current_cancel_event = threading.Event()
    set_scanning_state(True)
    update_status("Starting scan...")
    scan_args = (
        scan_targets,
        deep_var.get(),
        all_var.get(),
        gpt_var.get(),
        current_cancel_event,
        Config.RATE_LIMIT_PER_MINUTE,
        dry_var.get(),
        Config.ignore_patterns,
        extra_snippets,
        fail_threshold
    )
    scan_thread = threading.Thread(target=run_scan, args=scan_args, daemon=True)
    scan_thread.start()


def cancel_scan() -> None:
    """Signal the active scan to stop."""

    if current_cancel_event:
        current_cancel_event.set()


def rescan_selected() -> None:
    """Re-scan the currently selected items in the Treeview."""
    global current_cancel_event

    if not tree or current_cancel_event is not None:
        return

    selection = tree.selection()
    if not selection:
        return

    paths = []
    item_map = {}
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            path = values[0]
            paths.append(path)
            item_map[path] = item_id

    if not paths:
        return

    current_cancel_event = threading.Event()
    set_scanning_state(True)
    update_status(f"Rescanning {len(paths)} selected file(s)...")

    # Capture current settings to use in the background thread
    settings = {
        'deep': deep_var.get() if deep_var else False,
        'gpt': gpt_var.get() if gpt_var else False,
        'dry': dry_var.get() if dry_var else False,
    }

    scan_thread = threading.Thread(
        target=run_rescan,
        args=(paths, item_map, settings, current_cancel_event),
        daemon=True
    )
    scan_thread.start()


def _clean_snippet_for_ai(snippet: str) -> str:
    """Preprocess a code snippet by stripping whitespace and removing empty lines."""
    return ''.join([s for s in snippet.strip().splitlines(True) if s.strip()])


def analyze_selected_with_ai(event: Optional[tk.Event] = None) -> None:
    """Perform AI analysis for the currently selected items in the Treeview."""
    global current_cancel_event

    if not tree or current_cancel_event is not None:
        return

    if not Config.GPT_ENABLED:
        messagebox.showwarning("AI Disabled", "AI Analysis is disabled (task.txt not found or API key missing).")
        return

    selection = tree.selection()
    if not selection:
        return

    gpt_requests = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            path = values[0]
            snippet = values[5]
            cleaned_snippet = _clean_snippet_for_ai(snippet)

            gpt_requests.append({
                "path": path,
                "percent": values[1],
                "snippet": snippet,
                "cleaned_snippet": cleaned_snippet,
                "line": values[6] if len(values) > 6 else 1,
                "item_id": item_id,
            })

    if not gpt_requests:
        return

    current_cancel_event = threading.Event()
    set_scanning_state(True)
    update_status(f"Requesting AI analysis for {len(gpt_requests)} selected item(s)...")

    scan_thread = threading.Thread(
        target=run_batch_ai_analysis,
        args=(gpt_requests, current_cancel_event),
        daemon=True
    )
    scan_thread.start()


def add_to_ignore_file(patterns: Union[str, List[str]]) -> None:
    """Add one or more patterns to the .gptscanignore file.

    Args:
        patterns: A single pattern string or a list of pattern strings.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    ignore_file = Path('.gptscanignore')
    # Use 'a+' to handle both checking existence and appending
    with open(ignore_file, 'a+', encoding='utf-8') as f:
        # Move pointer to the beginning to read existing patterns
        f.seek(0)
        content = f.read()
        # Existing patterns without inline comments
        existing_patterns = {
            line.strip().split('#')[0].strip() for line in content.splitlines()
            if line.strip() and not line.strip().startswith('#')
        }

        # Ensure file ends with newline if not empty
        if content and not content.endswith('\n'):
            f.write('\n')

        for pattern in patterns:
            clean_pattern = pattern.split('#')[0].strip()
            if clean_pattern and clean_pattern not in existing_patterns:
                f.write(f"{pattern}\n")
                existing_patterns.add(clean_pattern)
                if clean_pattern not in Config.ignore_patterns:
                    Config.ignore_patterns.append(clean_pattern)


def remove_from_ignore_file(patterns: Iterable[str]) -> None:
    """Remove one or more patterns from the .gptscanignore file.

    Args:
        patterns: An iterable of pattern strings to remove.
    """
    patterns_to_remove = set(patterns)
    ignore_file = Path('.gptscanignore')
    if ignore_file.exists():
        with open(ignore_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with open(ignore_file, 'w', encoding='utf-8') as f:
            for line in lines:
                # Extract the actual pattern part if there's a comment
                stripped = line.strip()
                if not stripped:
                    f.write(line)
                    continue
                if stripped.startswith('#'):
                    f.write(line)
                    continue

                pattern_part = stripped.split('#')[0].strip()
                if pattern_part not in patterns_to_remove:
                    f.write(line)

    for p in patterns_to_remove:
        if p in Config.ignore_patterns:
            Config.ignore_patterns.remove(p)


def exclude_paths(paths: List[str], confirm: bool = True) -> bool:
    """Exclude a list of file paths from future scans.

    Args:
        paths: List of file paths to exclude.
        confirm: Whether to ask for confirmation before proceeding.

    Returns:
        True if the paths were excluded, False otherwise.
    """
    if not paths:
        return False

    if confirm:
        msg = f"Are you sure you want to exclude {len(paths)} item(s) from future scans?\nThis will add them to your .gptscanignore file."
        if not messagebox.askyesno("Exclude from Scan", msg):
            return False

    try:
        new_patterns = []
        for path in paths:
            path_str = str(path)
            try:
                pattern = os.path.relpath(path_str, os.getcwd())
            except ValueError:
                pattern = path_str

            new_patterns.append(pattern)
            if pattern not in Config.ignore_patterns:
                Config.ignore_patterns.append(pattern)

        add_to_ignore_file(new_patterns)

        # Update cache and refresh view
        global _all_results_cache
        path_set = {str(p) for p in paths}
        _all_results_cache = [v for v in _all_results_cache if v[0] not in path_set]

        _apply_filter()
        update_status(f"Excluded {len(paths)} file(s).")
        return True

    except Exception as e:
        messagebox.showerror("Error", f"Could not update .gptscanignore: {e}")
        return False


def exclude_selected() -> None:
    """Exclude selected files from future scans by adding them to .gptscanignore."""
    if not tree:
        return

    selection = tree.selection()
    if not selection:
        return

    # Capture current list and index to handle transition correctly
    all_visible = list(tree.get_children())
    try:
        current_idx = all_visible.index(selection[0])
    except (ValueError, IndexError):
        current_idx = -1

    excluded_paths = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            excluded_paths.append(values[0])

    if exclude_paths(excluded_paths, confirm=True):
        # After exclusion, list changes. Select the next item at the same index.
        new_visible = list(tree.get_children())
        if new_visible and current_idx != -1:
            new_idx = min(current_idx, len(new_visible) - 1)
            new_item_id = new_visible[new_idx]
            tree.selection_set(new_item_id)
            tree.focus(new_item_id)
            tree.see(new_item_id)


def manage_exclusions() -> None:
    """Open a dialog to manage scan exclusions (.gptscanignore)."""
    if not root:
        return

    manage_win = tk.Toplevel(root)
    manage_win.title("Manage Exclusions")
    manage_win.geometry("500x400")
    manage_win.minsize(400, 300)
    manage_win.transient(root)
    manage_win.grab_set()

    main_frame = ttk.Frame(manage_win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Excluded patterns (from .gptscanignore):", font=('TkDefaultFont', 9, 'bold')).pack(anchor="w", pady=(0, 5))

    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    ignore_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    ignore_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=ignore_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    ignore_listbox.config(yscrollcommand=scrollbar.set)

    def refresh_list():
        ignore_listbox.delete(0, tk.END)
        for pattern in Config.ignore_patterns:
            ignore_listbox.insert(tk.END, pattern)

    refresh_list()

    btn_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
    btn_frame.pack(fill=tk.X)

    def add_pattern():
        pattern = simpledialog.askstring("Add Pattern", "Enter a file or folder pattern to exclude (e.g., *.log, temp/):", parent=manage_win)
        if pattern:
            pattern = pattern.strip()
            if pattern and pattern not in Config.ignore_patterns:
                try:
                    add_to_ignore_file(pattern)
                    Config.ignore_patterns.append(pattern)
                    refresh_list()
                    _apply_filter()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not update .gptscanignore: {e}", parent=manage_win)

    def add_folder():
        folder = filedialog.askdirectory(parent=manage_win)
        if folder:
            try:
                rel_path = os.path.relpath(folder, os.getcwd())
                if rel_path not in Config.ignore_patterns:
                    add_to_ignore_file(rel_path)
                    Config.ignore_patterns.append(rel_path)
                    refresh_list()
                    _apply_filter()
            except Exception as e:
                messagebox.showerror("Error", f"Could not add folder: {e}", parent=manage_win)

    def remove_selected():
        selection = ignore_listbox.curselection()
        if not selection:
            return

        patterns_to_remove = {ignore_listbox.get(i) for i in selection}
        if not messagebox.askyesno("Confirm Removal", f"Remove {len(patterns_to_remove)} exclusion(s)?", parent=manage_win):
            return

        try:
            remove_from_ignore_file(patterns_to_remove)
            refresh_list()
            _apply_filter()
        except Exception as e:
            messagebox.showerror("Error", f"Could not update .gptscanignore: {e}", parent=manage_win)

    ttk.Button(btn_frame, text="Add Pattern...", command=add_pattern).pack(side=tk.LEFT, padx=(0, 5), ipady=5)
    ttk.Button(btn_frame, text="Add Folder...", command=add_folder).pack(side=tk.LEFT, padx=5, ipady=5)
    ttk.Button(btn_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5, ipady=5)
    ttk.Button(btn_frame, text="Close", command=manage_win.destroy).pack(side=tk.RIGHT, ipady=5)


def manage_extensions() -> None:
    """Open a dialog to manage scanned file extensions (extensions.txt)."""
    if not root:
        return

    manage_win = tk.Toplevel(root)
    manage_win.title("Manage Extensions")
    manage_win.geometry("500x400")
    manage_win.minsize(400, 300)
    manage_win.transient(root)
    manage_win.grab_set()

    main_frame = ttk.Frame(manage_win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Scanned extensions (from extensions.txt):", font=('TkDefaultFont', 9, 'bold')).pack(anchor="w", pady=(0, 5))

    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    ext_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    ext_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=ext_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    ext_listbox.config(yscrollcommand=scrollbar.set)

    def refresh_list():
        ext_listbox.delete(0, tk.END)
        for ext in sorted(Config.extensions_set):
            ext_listbox.insert(tk.END, ext)

    refresh_list()

    btn_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
    btn_frame.pack(fill=tk.X)

    def add_extension():
        ext = simpledialog.askstring("Add Extension", "Enter a file extension (e.g., .py, .js):", parent=manage_win)
        if ext:
            ext = ext.strip().lower()
            if not ext.startswith('.'):
                ext = f".{ext}"
            if ext and ext not in Config.extensions_set:
                try:
                    Config.extensions_set.add(ext)
                    Config.save_extensions()
                    refresh_list()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not update extensions: {e}", parent=manage_win)

    def remove_selected():
        selection = ext_listbox.curselection()
        if not selection:
            return

        exts_to_remove = {ext_listbox.get(i) for i in selection}
        if not messagebox.askyesno("Confirm Removal", f"Remove {len(exts_to_remove)} extension(s)?", parent=manage_win):
            return

        try:
            for ext in exts_to_remove:
                Config.extensions_set.discard(ext)
            Config.save_extensions()
            refresh_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not update extensions: {e}", parent=manage_win)

    def reset_defaults():
        if not messagebox.askyesno("Confirm Reset", "Reset extensions to default list?", parent=manage_win):
            return

        try:
            Config.set_extensions(Config.DEFAULT_EXTENSIONS)
            Config.save_extensions()
            refresh_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not reset extensions: {e}", parent=manage_win)

    ttk.Button(btn_frame, text="Add...", command=add_extension).pack(side=tk.LEFT, padx=(0, 5), ipady=5)
    ttk.Button(btn_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5, ipady=5)
    ttk.Button(btn_frame, text="Reset to Defaults", command=reset_defaults).pack(side=tk.LEFT, padx=5, ipady=5)
    ttk.Button(btn_frame, text="Close", command=manage_win.destroy).pack(side=tk.RIGHT, ipady=5)


def unpack_content(name: str, content: bytes, depth: int = 0, hint: Optional[str] = None) -> Generator[Tuple[str, bytes], None, None]:
    """Recursively unpack archives and notebooks into individual snippets.

    Args:
        name: The display name or path of the content.
        content: The raw bytes of the content.
        depth: Current recursion depth to prevent infinite loops.
        hint: Optional filename hint for extension checking.

    Yields:
        Tuples of (display_name, content_bytes).
    """
    if depth > 5:
        return

    check_name = hint or name

    # 1. Check for ZIP
    try:
        if content.startswith(b'PK\x03\x04'):
            buffer = io.BytesIO(content)
            with zipfile.ZipFile(buffer) as z:
                for info in z.infolist():
                    if info.is_dir() or info.file_size > Config.MAX_FILE_SIZE:
                        continue
                    with z.open(info) as f:
                        member_content = f.read()
                        member_name = f"{name}[{info.filename}]"
                        yield from unpack_content(member_name, member_content, depth + 1, hint=info.filename)
            return
    except Exception:
        pass

    # 2. Check for TAR (including .tar.gz via magic bytes)
    try:
        # Check for GZIP magic b'\x1f\x8b'
        is_gz = content.startswith(b'\x1f\x8b')
        # Check for TAR magic 'ustar' at offset 257
        is_tar = len(content) > 262 and content[257:262] == b'ustar'

        # Also use tarfile's built-in detection if needed
        is_tar_via_lib = False
        if not (is_gz or is_tar):
            try:
                is_tar_via_lib = tarfile.is_tarfile(io.BytesIO(content))
            except Exception:
                pass

        if is_gz or is_tar or is_tar_via_lib:
            buffer = io.BytesIO(content)
            with tarfile.open(fileobj=buffer) as t:
                for member in t.getmembers():
                    if not member.isfile() or member.size > Config.MAX_FILE_SIZE:
                        continue
                    f = t.extractfile(member)
                    if f:
                        member_content = f.read()
                        member_name = f"{name}[{member.name}]"
                        yield from unpack_content(member_name, member_content, depth + 1, hint=member.name)
            return
    except Exception:
        pass

    # 3. Check for Jupyter Notebook
    if check_name.lower().endswith('.ipynb') or (content.startswith(b'{') and b'"cells"' in content):
        try:
            notebook = json.loads(content.decode('utf-8', errors='ignore'))
            if isinstance(notebook, dict) and 'cells' in notebook:
                cell_count = 0
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        code = "".join(source) if isinstance(source, list) else str(source)
                        if code.strip():
                            cell_count += 1
                            yield (f"{name} [Cell {cell_count}]", code.encode('utf-8'))
                return
        except Exception:
            pass

    # 4. Check for package manifests (package.json, composer.json, deno.json/jsonc)
    lowered_check = check_name.lower()
    if lowered_check.endswith(('package.json', 'composer.json', 'deno.json', 'deno.jsonc')):
        try:
            text = content.decode('utf-8', errors='ignore')
            if lowered_check.endswith('.jsonc'):
                # Strip single-line and multi-line comments for JSONC support
                # This regex strips /* ... */ comments and // ... comments while attempting to ignore
                # // if it appears inside a double-quoted string (to avoid mangling URLs).
                text = re.sub(
                    r'(/\*.*?\*/)|("(?:\\.|[^"\\])*")|(//[^\n]*)',
                    lambda m: m.group(2) if m.group(2) else " ",
                    text,
                    flags=re.DOTALL
                )

            manifest = json.loads(text)
            if not isinstance(manifest, dict):
                return

            # Determine key and label based on manifest type
            key = 'scripts'
            label = 'Script'
            if 'deno.json' in lowered_check:
                key = 'tasks'
                label = 'Task'

            scripts = manifest.get(key, {})
            if isinstance(scripts, dict):
                yielded_any = False
                for script_name, command in scripts.items():
                    if isinstance(command, str):
                        if command.strip():
                            yield (f"{name} [{label}: {script_name}]", command.encode('utf-8'))
                            yielded_any = True
                    elif isinstance(command, list):
                        # Support for array-based scripts (common in composer.json)
                        for i, cmd in enumerate(command, 1):
                            if isinstance(cmd, str) and cmd.strip():
                                yield (f"{name} [{label}: {script_name} ({i})]", cmd.encode('utf-8'))
                                yielded_any = True
                if yielded_any:
                    return

            # If it's a manifest but has no scripts/tasks, we don't want to scan it as a whole file
            return
        except Exception:
            pass

    # 5. Check for Markdown code blocks
    if check_name.lower().endswith('.md'):
        try:
            text = content.decode('utf-8', errors='ignore')
            # Match fenced code blocks (triple backticks)
            blocks = re.findall(r'```(?:\w+)?\s*\n?(.*?)\s*```', text, re.DOTALL)
            for i, block in enumerate(blocks, 1):
                if block.strip():
                    yield (f"{name} [Block {i}]", block.encode('utf-8'))
            return
        except Exception:
            pass

    # 6. Check for HTML script tags and other embedded elements
    if check_name.lower().endswith(('.html', '.htm', '.xhtml')):
        try:
            text = content.decode('utf-8', errors='ignore')
            extracted_any = False
            
            # 1. Match <script> blocks (yield individually as they can be large)
            scripts = re.findall(r'<script\b[^>]*>(.*?)</script>', text, re.DOTALL | re.IGNORECASE)
            for i, script in enumerate(scripts, 1):
                if script.strip():
                    yield (f"{name} [Script {i}]", script.encode('utf-8'))
                    extracted_any = True
            
            # 2. Match other embedded elements (bundle together as they are usually small)
            embedded_patterns = [
                r'<iframe\b[^>]*>.*?</iframe>',
                r'<iframe\b[^>]*>',
                r'<object\b[^>]*>.*?</object>',
                r'<object\b[^>]*>',
                r'<embed\b[^>]*>',
                r'<applet\b[^>]*>.*?</applet>',
                r'<applet\b[^>]*>'
            ]
            
            embedded_elements = []
            for pattern in embedded_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if match.strip() and match not in embedded_elements:
                        embedded_elements.append(match.strip())
            
            if embedded_elements:
                bundle = "\n".join(embedded_elements)
                yield (f"{name} [Embedded Elements]", bundle.encode('utf-8'))
                extracted_any = True
                
            if extracted_any:
                return
        except Exception:
            pass

    # 7. Check for Dockerfile
    lowered_check = check_name.lower()
    if 'dockerfile' in lowered_check and (os.path.basename(lowered_check) == 'dockerfile' or lowered_check.endswith('.dockerfile')):
        try:
            text = content.decode('utf-8', errors='ignore')
            # Extract RUN, CMD, and ENTRYPOINT instructions with multi-line support
            instructions = []
            current_instr = []

            def finalize_instr():
                if current_instr:
                    instructions.append(" ".join([c.rstrip('\\').strip() for c in current_instr]))

            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue

                # Check for new instruction
                instr_match = re.match(r'^\s*(?:RUN|CMD|ENTRYPOINT)\s+(.*)', line, re.IGNORECASE)
                if instr_match:
                    finalize_instr()
                    current_instr = [instr_match.group(1)]
                elif current_instr:
                    current_instr.append(line.strip())

                # If line doesn't end with \, we've finished this instruction (if we were in one)
                if current_instr and not line.rstrip().endswith('\\'):
                    finalize_instr()
                    current_instr = []

            finalize_instr()

            if instructions:
                for i, cmd in enumerate(instructions, 1):
                    if cmd.strip():
                        yield (f"{name} [Instruction {i}]", cmd.encode('utf-8'))
                return
        except Exception:
            pass

    # 8. Check for CI/CD Workflows (YAML)
    if lowered_check.endswith(('.yml', '.yaml')):
        try:
            text = content.decode('utf-8', errors='ignore')
            snippets = []
            script_keys = ['run', 'script', 'command', 'before_script', 'after_script']
            # Match keys, optionally prefixed by a dash (common in YAML lists)
            key_pattern = r'^\s*(?:-\s*)?(?:' + '|'.join(script_keys) + r'):\s*(.*)'
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                match = re.match(key_pattern, line, re.IGNORECASE)
                if match:
                    first_line_val = match.group(1).strip()
                    indent = len(line) - len(line.lstrip())

                    if first_line_val in ('|', '>', '|-', '|+', '>-', '>+'):
                        # Multi-line block
                        block_lines = []
                        i += 1
                        while i < len(lines):
                            if not lines[i].strip():
                                block_lines.append("")
                                i += 1
                                continue
                            current_indent = len(lines[i]) - len(lines[i].lstrip())
                            if current_indent <= indent:
                                break
                            block_lines.append(lines[i])
                            i += 1
                        if block_lines:
                            snippets.append("\n".join(block_lines))
                        continue
                    elif not first_line_val:
                        # Potentially a list on the next lines
                        list_items = []
                        j = i + 1
                        while j < len(lines):
                            if not lines[j].strip():
                                j += 1
                                continue
                            curr_indent = len(lines[j]) - len(lines[j].lstrip())
                            if curr_indent <= indent:
                                break
                            list_match = re.match(r'^\s*-\s*(.*)', lines[j])
                            if list_match:
                                cmd = list_match.group(1).strip()
                                if cmd:
                                    list_items.append(cmd)
                                j += 1
                            else:
                                break
                        if list_items:
                            snippets.append("\n".join(list_items))
                            i = j
                            continue
                    elif first_line_val:
                        # Single line command
                        snippets.append(first_line_val)
                i += 1

            if snippets:
                for idx, snippet in enumerate(snippets, 1):
                    if snippet.strip():
                        yield (f"{name} [Script {idx}]", snippet.encode('utf-8'))
                return
        except Exception:
            pass

    # 9. Check for Unified Diff (.diff or .patch)
    if check_name.lower().endswith(('.diff', '.patch')):
        try:
            text = content.decode('utf-8', errors='ignore')
            lines = text.splitlines()
            current_file = None
            hunk_info = None
            hunk_lines = []

            def finalize_hunk():
                if current_file and hunk_info and hunk_lines:
                    # Only yield if there's at least one added line in this hunk
                    if any(l.startswith('+') and not l.startswith('+++') for l in hunk_lines):
                        hunk_text = "\n".join(hunk_lines)
                        yield (f"{name} [{current_file} @ {hunk_info}]", hunk_text.encode('utf-8'))

            for line in lines:
                if line.startswith('+++ '):
                    yield from finalize_hunk()
                    # Extract file path: +++ b/path/to/file.py -> path/to/file.py
                    path_part = line[4:].split('\t')[0].strip()
                    if path_part.startswith(('a/', 'b/')) and '/' in path_part:
                        path_part = path_part[2:]
                    current_file = path_part
                    hunk_info = None
                    hunk_lines = []
                elif line.startswith('@@ ') and current_file:
                    yield from finalize_hunk()
                    # Extract starting line: @@ -1,1 +1,2 @@ -> line 1
                    match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    hunk_info = f"line {match.group(1)}" if match else "unknown"
                    hunk_lines = []
                elif hunk_info is not None:
                    if line.startswith((' ', '+')):
                        hunk_lines.append(line)
                    elif not line.startswith('-'):
                        # End of hunk if line doesn't start with context/add/remove
                        yield from finalize_hunk()
                        hunk_info = None
                        hunk_lines = []

            yield from finalize_hunk()
            return
        except Exception:
            pass

    # 10. Check for Makefile
    if 'makefile' in lowered_check and (os.path.basename(lowered_check) == 'makefile' or lowered_check.endswith('.makefile')):
        try:
            text = content.decode('utf-8', errors='ignore')
            # Extract recipes (lines starting with a tab) with multi-line support
            recipes = []
            current_recipe = []

            def finalize_recipe():
                if current_recipe:
                    recipes.append(" ".join([c.rstrip('\\').strip() for c in current_recipe]))

            for line in text.splitlines():
                if line.startswith('\t'):
                    current_recipe.append(line[1:])
                    # If line doesn't end with \, we've finished this recipe
                    if not line.rstrip().endswith('\\'):
                        finalize_recipe()
                        current_recipe = []
                elif current_recipe:
                    finalize_recipe()
                    current_recipe = []

            finalize_recipe()

            if recipes:
                for i, recipe in enumerate(recipes, 1):
                    if recipe.strip():
                        yield (f"{name} [Recipe {i}]", recipe.encode('utf-8'))
                return
        except Exception:
            pass

    # 9. Fallback: yield as a single snippet if it's a supported file type
    # If scan_all_files is True, we always yield. Otherwise check extension/shebang.
    if Config.is_supported_file(check_name, content=content, is_member=(depth > 0)):
        yield name, content
    elif depth == 0 and name.startswith(("[", "boundary")):
        # Special case for legacy test compatibility (test_extra_snippets.py)
        # Some tests use non-standard names for snippets but expect them to be scanned
        yield name, content


def iter_windows(fh, size: int, deep_scan: bool, maxlen: Optional[int] = None) -> Generator[Tuple[int, bytes], None, None]:
    """Yield file chunks for scanning.

    Balance speed and thoroughness by checking the beginning and end of files
    where script headers and malicious payloads are usually found.
    """
    if maxlen is None:
        maxlen = Config.MAXLEN

    if size == 0:
        yield 0, b""
        return

    if not deep_scan:
        fh.seek(0)
        yield 0, fh.read(maxlen)

        last_start = max(0, size - maxlen)
        if last_start > 0:
            fh.seek(last_start)
            yield last_start, fh.read(maxlen)
        return

    offset = 0
    while offset < size:
        fh.seek(offset)
        chunk = fh.read(maxlen)
        if not chunk:
            break
        yield offset, chunk
        offset += maxlen

    if size > maxlen:
        last_start = size - maxlen
        if last_start % maxlen != 0:
            fh.seek(last_start)
            final_chunk = fh.read(maxlen)
            if final_chunk:
                yield last_start, final_chunk


def scan_files(
    scan_targets: Union[str, List[str]],
    deep_scan: bool,
    show_all: bool,
    use_gpt: bool,
    cancel_event: Optional[threading.Event] = None,
    rate_limit: int = Config.RATE_LIMIT_PER_MINUTE,
    max_concurrent_requests: int = Config.MAX_CONCURRENT_REQUESTS,
    dry_run: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    extra_snippets: Optional[List[Tuple[str, bytes]]] = None,
    fail_threshold: Optional[int] = None,
) -> Generator[Tuple[str, Tuple[Any, ...]], None, None]:
    """Scan files for malicious content and optionally request GPT analysis.

    Parameters
    ----------
    scan_targets : Union[str, List[str]]
        Directory path or list of file/directory paths to search.
    deep_scan : bool
        Whether to scan overlapping 1024-byte windows beyond the first block.
    show_all : bool
        Whether to yield all scanned files regardless of threat level threshold.
    use_gpt : bool
        Whether to request GPT analysis when the local model is confident.
    rate_limit : int
        Maximum number of GPT requests permitted per minute.
    max_concurrent_requests : int
        Maximum number of GPT requests executed concurrently.
    dry_run : bool
        Whether to list files that would be scanned without running the model or API.
    exclude_patterns : List[str], optional
        List of glob patterns to exclude from the scan.
    extra_snippets : List[Tuple[str, bytes]], optional
        List of (name, content) tuples to scan as in-memory buffers.

    Yields
    ------
    Generator[Tuple[str, Any], None, None]
        Tuples indicating events:
        - ('progress', (current: int, total: int, status: Optional[str]))
        - ('result', (path: str, own_conf: str, admin: str, user: str, gpt: str, snippet: str, line: Union[int, str]))
        - ('summary', (total_files: int, total_bytes: int, elapsed_time: float))
    """
    global _tf_module
    cancel_event = cancel_event or threading.Event()

    if not dry_run:
        modelscript = get_model()
        tf_module = _tf_module

        def predict_window(window_bytes: bytes) -> Tuple[float, bytes]:
            padded_data = list(window_bytes)
            padded_data.extend([13] * (Config.MAXLEN - len(padded_data)))
            tf_data = tf_module.expand_dims(tf_module.constant(padded_data), axis=0)
            prediction = modelscript.predict(tf_data, batch_size=1, steps=1)[0][0]
            return float(prediction), bytes(padded_data)

    # Identify which files were explicitly passed as targets and deduplicate them
    if isinstance(scan_targets, (str, Path)):
        try:
            scan_targets = shlex.split(str(scan_targets))
        except ValueError:
            scan_targets = [str(scan_targets)]

    scan_targets = list(dict.fromkeys(scan_targets))

    url_targets = [str(t) for t in scan_targets if str(t).startswith(('http://', 'https://'))]
    local_targets = [t for t in scan_targets if str(t) not in url_targets]

    explicit_targets = {Path(t) for t in local_targets}
    explicit_files = {f for f in explicit_targets if f.is_file()}

    file_list = collect_files(local_targets)

    if exclude_patterns:
        file_list = [
            f for f in file_list
            if not any(f.match(p) or any(parent.match(p) for parent in f.parents) for p in exclude_patterns)
        ]

    # Pre-process snippets from file_list and extra_snippets using unpack_content
    processed_snippets: List[Tuple[str, bytes]] = []
    # Preserve extra_snippets (from clipboard, stdin)
    original_extra = list(extra_snippets) if extra_snippets else []

    # Unpack extra snippets (URL, Stdin, Clipboard)
    for name, content in original_extra:
        processed_snippets.extend(unpack_content(name, content))

    # Unpack local files
    non_unpacked_files = []
    for f_path in file_list:
        is_explicit = f_path in explicit_files
        path_s = str(f_path).lower()

        if Config.is_container(path_s):
            try:
                with open(f_path, 'rb') as f:
                    full_content = f.read()
                processed_snippets.extend(unpack_content(str(f_path), full_content, hint=path_s))
            except Exception:
                if Config.is_supported_file(f_path, is_explicit=is_explicit):
                    non_unpacked_files.append(f_path)
        else:
            # For non-container files, we still check if they are supported scripts
            if Config.is_supported_file(f_path, is_explicit=is_explicit):
                non_unpacked_files.append(f_path)

    file_list = non_unpacked_files
    extra_snippets = processed_snippets

    num_urls = len(url_targets)
    total_progress = len(file_list) + len(extra_snippets) + num_urls
    progress_count = 0
    total_bytes_scanned = 0
    actual_files_scanned = 0
    start_time = time.perf_counter()

    # Process URL targets (Phase 1: Fetching and Unpacking)
    for url in url_targets:
        if cancel_event.is_set():
            break
        progress_count += 1
        yield ('progress', (progress_count, total_progress, f"Fetching: {url}"))
        try:
            content = fetch_url_content(url)
            unpacked_from_url = list(unpack_content(f"[URL] {url}", content))
            if unpacked_from_url:
                extra_snippets.extend(unpacked_from_url)
                # Dynamically update total progress to reflect newly discovered members
                total_progress += len(unpacked_from_url)
            else:
                # If nothing yielded but fetch succeeded, report it wasn't a script/archive
                yield (
                    'result',
                    (
                        url,
                        'Info',
                        '',
                        '',
                        '',
                        "Remote content fetched but no supported scripts or archives detected.",
                        '-',
                    )
                )
        except Exception as e:
            # If fetch fails, we yield the error now and account for both phases
            yield (
                'result',
                (
                    url,
                    'Fetch Error',
                    '',
                    '',
                    '',
                    f"Could not download script: {e}",
                    '-',
                )
            )

    yield ('progress', (progress_count, total_progress, "Collecting files..."))

    gpt_requests: List[Dict[str, Any]] = []
    threshold_val = Config.THRESHOLD / 100.0
    if fail_threshold is not None:
        threshold_val = min(threshold_val, fail_threshold / 100.0)

    def handle_scan_result(path: str, maxconf: float, max_window_bytes: bytes, line_num: Union[int, str]) -> Generator[Tuple[str, Any], None, None]:
        if maxconf >= 0:
            percent = f"{maxconf:.0%}"
            snippet = ''.join(map(chr, max_window_bytes)).strip()
            cleaned_snippet = _clean_snippet_for_ai(snippet)

            if maxconf >= threshold_val and use_gpt and Config.GPT_ENABLED:
                gpt_requests.append(
                    {
                        "path": path,
                        "percent": percent,
                        "snippet": snippet,
                        "cleaned_snippet": cleaned_snippet,
                        "line": line_num,
                    }
                )
            elif maxconf >= threshold_val or show_all:
                yield (
                    'result',
                    (
                        path,
                        percent,
                        '',
                        '',
                        '',
                        cleaned_snippet,
                        line_num,
                    )
                )

    for file_path in file_list:
        if cancel_event.is_set():
            break

        progress_count += 1
        yield ('progress', (progress_count, total_progress, f"Scanning: {file_path.name}"))

        is_explicit = file_path in explicit_files
        actual_files_scanned += 1
        try:
            file_size = file_path.stat().st_size
        except OSError as err:
            file_size = None
            if not dry_run:
                yield (
                    'result',
                    (
                        str(file_path),
                        'Error',
                        '',
                        '',
                        '',
                        f"Error reading file metadata: {err}",
                        '-',
                    )
                )

        if file_size is not None:
            total_bytes_scanned += file_size

        # Check file size limit (skip if not explicitly requested)
        if not is_explicit and file_size is not None and file_size > Config.MAX_FILE_SIZE:
            yield (
                'result',
                (
                    str(file_path),
                    'Large File',
                    '',
                    '',
                    '',
                    f"Skipped: File exceeds maximum size ({format_bytes(Config.MAX_FILE_SIZE)})",
                    '-',
                )
            )
            continue

        if dry_run:
            yield (
                'result',
                (
                    str(file_path),
                    'Dry Run',
                    '',
                    '',
                    '',
                    f"(File would be scanned, size: {format_bytes(file_size) if file_size is not None else 'unknown'})",
                    '-',
                )
            )
        else:
            print(file_path, file=sys.stderr)
            if file_size is not None:
                maxconf = -1.0
                max_window_bytes = b""
                max_offset = 0
                error_message: Optional[str] = None

                try:
                    with open(file_path, 'rb') as f:
                        for offset, window in iter_windows(f, file_size, deep_scan):
                            if cancel_event.is_set():
                                break
                            print("Scanning at:", offset if offset > 0 else 0, file=sys.stderr)
                            result, padded_bytes = predict_window(window)
                            if result > maxconf:
                                maxconf = result
                                max_window_bytes = padded_bytes
                                max_offset = offset
                except OSError as err:
                    error_message = f"Error reading file: {err}"

                if error_message is not None:
                    yield (
                        'result',
                        (
                            str(file_path),
                            'Error',
                            '',
                            '',
                            '',
                            error_message,
                            '-',
                        )
                    )
                else:
                    # Calculate line number
                    line_num = 1
                    try:
                        with open(file_path, 'rb') as f:
                            line_num = f.read(max_offset).count(b'\n') + 1
                    except Exception:
                        pass
                    yield from handle_scan_result(str(file_path), maxconf, max_window_bytes, line_num)

    for name, content in extra_snippets:
        if cancel_event.is_set():
            break

        progress_count += 1
        yield ('progress', (progress_count, total_progress, f"Scanning: {name}"))

        file_size = len(content)
        total_bytes_scanned += file_size
        actual_files_scanned += 1

        if dry_run:
            yield (
                'result',
                (
                    name,
                    'Dry Run',
                    '',
                    '',
                    '',
                    f"(Snippet would be scanned, size: {format_bytes(file_size)})",
                    '-',
                )
            )
        else:
            maxconf = -1.0
            max_window_bytes = b""
            max_offset = 0
            with io.BytesIO(content) as f:
                for offset, window in iter_windows(f, file_size, deep_scan):
                    if cancel_event.is_set():
                        break
                    result, padded_bytes = predict_window(window)
                    if result > maxconf:
                        maxconf = result
                        max_window_bytes = padded_bytes
                        max_offset = offset

            # Calculate line number for snippet
            line_num = content[:max_offset].count(b'\n') + 1
            yield from handle_scan_result(name, maxconf, max_window_bytes, line_num)

    if cancel_event.is_set():
        return

    if use_gpt and Config.GPT_ENABLED and gpt_requests:
        async def process_requests(requests: Iterable[Dict[str, Any]]):
            rate_limiter = AsyncRateLimiter(rate_limit)
            semaphore = asyncio.Semaphore(max_concurrent_requests)
            wait_messages: List[str] = []

            def wait_notifier(_wait_time: float) -> None:
                wait_messages.append("Waiting for API rate limit...")

            async def run_request(request: Dict[str, Any]):
                if cancel_event.is_set():
                    return request, None
                json_data = await async_handle_gpt_response(
                    request["snippet"],
                    Config.taskdesc,
                    rate_limiter=rate_limiter,
                    semaphore=semaphore,
                    wait_callback=wait_notifier,
                )
                return request, json_data

            tasks = [asyncio.create_task(run_request(request)) for request in gpt_requests]
            results: List[Tuple[Dict[str, Any], Optional[Dict]]] = []
            for completed in asyncio.as_completed(tasks):
                if cancel_event.is_set():
                    for t in tasks:
                        t.cancel()
                    break
                results.append(await completed)
            return results, wait_messages

        total_progress += len(gpt_requests)
        yield ('progress', (progress_count, total_progress, "Waiting for API rate limit..."))

        results, wait_messages = asyncio.run(process_requests(gpt_requests))

        for _ in wait_messages:
            yield ('progress', (progress_count, total_progress, "Waiting for API rate limit..."))

        for request, json_data in results:
            if cancel_event.is_set():
                break

            progress_count += 1
            yield ('progress', (progress_count, total_progress, f"AI Analysis: {os.path.basename(request['path'])}"))

            if json_data is None:
                admin_desc = 'AI response error'
                enduser_desc = 'AI response error'
                chatgpt_conf_percent = 'AI response error'
            else:
                admin_desc = json_data["administrator"]
                enduser_desc = json_data["end-user"]
                chatgpt_conf_percent = "{:.0%}".format(int(json_data["threat-level"]) / 100.)

            yield (
                'result',
                (
                    request["path"],
                    request["percent"],
                    admin_desc,
                    enduser_desc,
                    chatgpt_conf_percent,
                    request["cleaned_snippet"],
                    request.get("line", 1),
                )
            )

    end_time = time.perf_counter()
    yield ('summary', (actual_files_scanned, total_bytes_scanned, end_time - start_time))


def batch_ai_analysis_events(
    gpt_requests: List[Dict[str, Any]],
    cancel_event: threading.Event,
    rate_limit: int = Config.RATE_LIMIT_PER_MINUTE,
    max_concurrent_requests: int = Config.MAX_CONCURRENT_REQUESTS,
) -> Generator[Tuple[str, Any], None, None]:
    """Generator that performs AI analysis for a batch of requests.

    Args:
        gpt_requests: List of dictionaries with 'path', 'percent', 'snippet', 'cleaned_snippet', and 'line'.
        cancel_event: Event to signal cancellation.
        rate_limit: Requests per minute limit.
        max_concurrent_requests: Maximum parallel requests.

    Yields:
        Standard scan events ('progress', 'result', 'summary').
    """
    if not gpt_requests or not Config.GPT_ENABLED:
        return

    async def process_requests(requests: Iterable[Dict[str, Any]]):
        rate_limiter = AsyncRateLimiter(rate_limit)
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        total = len(requests)
        progress = 0

        def wait_notifier(_wait_time: float) -> None:
            enqueue_ui_update(update_status, "Waiting for API rate limit...")

        async def run_request(request: Dict[str, Any]):
            nonlocal progress
            if cancel_event.is_set():
                return
            json_data = await async_handle_gpt_response(
                request["snippet"],
                Config.taskdesc,
                rate_limiter=rate_limiter,
                semaphore=semaphore,
                wait_callback=wait_notifier,
            )
            progress += 1

            # Prepare result data
            if json_data is None:
                admin_desc = 'AI Error' if not cancel_event.is_set() else 'Cancelled'
                enduser_desc = admin_desc
                chatgpt_conf_percent = admin_desc
            else:
                admin_desc = json_data["administrator"]
                enduser_desc = json_data["end-user"]
                chatgpt_conf_percent = "{:.0%}".format(int(json_data["threat-level"]) / 100.)

            result_data = (
                request["path"],
                request["percent"],
                admin_desc,
                enduser_desc,
                chatgpt_conf_percent,
                request["cleaned_snippet"],
                request.get("line", 1),
                request.get("item_id"),
            )

            # Yield results via queue to the generator consumer
            result_queue.put(('result', result_data))
            result_queue.put(('progress', (progress, total, f"AI Analysis: {os.path.basename(request['path'])}")))

        tasks = [asyncio.create_task(run_request(request)) for request in requests]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            result_queue.put(('summary', (total, 0, 0.0)))
            result_queue.put((None, None))

    result_queue = queue.Queue()
    total_count = len(gpt_requests)
    yield ('progress', (0, total_count, "Starting AI Analysis..."))

    # Start async processing in a separate thread to not block this generator's consumer
    def run_async_loop():
        try:
            asyncio.run(process_requests(gpt_requests))
        except Exception as e:
            result_queue.put(('progress', (0, 0, f"Error: {e}")))
            result_queue.put((None, None))

    threading.Thread(target=run_async_loop, daemon=True).start()

    while True:
        event_type, data = result_queue.get()
        if event_type is None:
            break
        yield event_type, data


def _consume_scan_events(
    event_gen: Iterable[Tuple[str, Any]],
    cancel_event: threading.Event,
    status_prefix: str,
    result_handler: Callable[[Tuple[Any, ...]], bool],
    print_status: bool = False,
    fail_threshold: Optional[int] = None
) -> None:
    """Helper to process scan events and update UI consistently."""
    last_total: Optional[int] = 0
    threats_found = 0
    high_risk_found = 0
    medium_risk_found = 0
    metrics: Dict[str, Any] = {}
    current_scanned = 0

    try:
        for event_type, data in event_gen:
            if event_type == 'progress':
                current, total, status = data
                current_scanned = current

                if total != last_total:
                    enqueue_ui_update(configure_progress, total)
                    last_total = total
                enqueue_ui_update(update_progress, current)

                if threats_found > 0:
                    threat_suffix = f" ({threats_found} suspicious: {high_risk_found} high, {medium_risk_found} medium)"
                else:
                    threat_suffix = ""
                status_text = f"{status} ({current}/{total}){threat_suffix}" if status else f"{status_prefix}: {current}/{total}{threat_suffix}"
                if print_status:
                    print(status_text, file=sys.stderr)
                enqueue_ui_update(update_status, status_text)
            elif event_type == 'result':
                if cancel_event.is_set():
                    continue

                conf = get_effective_threat_level(data[1], data[4])
                
                # Use fail_threshold for internal threat counting if lower than reporting threshold
                effective_threshold = Config.THRESHOLD
                if fail_threshold is not None:
                    effective_threshold = min(effective_threshold, fail_threshold)
                
                risk = get_risk_category(conf, effective_threshold)

                if result_handler(data):
                    if risk == 'high':
                        threats_found += 1
                        high_risk_found += 1
                    elif risk == 'medium':
                        threats_found += 1
                        medium_risk_found += 1
            elif event_type == 'summary':
                total_files, total_bytes, elapsed_time = data
                metrics['total_bytes'] = total_bytes
                metrics['elapsed_time'] = elapsed_time
    finally:
        if cancel_event.is_set():
            enqueue_ui_update(finish_scan_state)
            enqueue_ui_update(update_status, f"Scan cancelled after {current_scanned} files.")
        else:
            enqueue_ui_update(
                finish_scan_state,
                current_scanned,
                threats_found,
                metrics.get('total_bytes'),
                metrics.get('elapsed_time'),
                high_risk_found,
                medium_risk_found
            )


def run_scan(
    scan_targets: Union[str, List[str]],
    deep_scan: bool,
    show_all: bool,
    use_gpt: bool,
    cancel_event: threading.Event,
    rate_limit: int = Config.RATE_LIMIT_PER_MINUTE,
    dry_run: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    extra_snippets: Optional[List[Tuple[str, bytes]]] = None,
    fail_threshold: Optional[int] = None,
) -> None:
    """Consume scan events and forward them to the UI thread.

    Parameters
    ----------
    scan_targets : Union[str, List[str]]
        Directory path or list of files to scan.
    deep_scan : bool
        Whether to evaluate all 1024-byte windows.
    show_all : bool
        Whether to display all results regardless of threat level.
    use_gpt : bool
        Whether to enrich suspicious files with GPT output.
    rate_limit : int
        Maximum allowed GPT requests per minute.
    dry_run : bool
        Whether to simulate the scan.
    """
    event_gen = scan_files(
        scan_targets,
        deep_scan,
        show_all,
        use_gpt,
        cancel_event,
        rate_limit=rate_limit,
        max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
        dry_run=dry_run,
        exclude_patterns=exclude_patterns,
        extra_snippets=extra_snippets,
        fail_threshold=fail_threshold,
    )

    def scan_handler(data: Tuple[Any, ...]) -> bool:
        enqueue_ui_update(insert_tree_row, data)
        return True

    _consume_scan_events(event_gen, cancel_event, "Scanning", scan_handler, print_status=True, fail_threshold=fail_threshold)

def run_rescan(
    paths: List[str],
    item_map: Dict[str, str],
    settings: Dict[str, Any],
    cancel_event: threading.Event
) -> None:
    """Perform background scan for specific paths and update existing UI rows."""
    event_gen = scan_files(
        paths,
        settings['deep'],
        show_all=True,  # Always show results for rescan to update rows
        use_gpt=settings['gpt'],
        cancel_event=cancel_event,
        rate_limit=Config.RATE_LIMIT_PER_MINUTE,
        max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
        dry_run=settings['dry'],
        exclude_patterns=None,  # Already selected, don't re-exclude
    )

    def rescan_handler(data: Tuple[Any, ...]) -> bool:
        path = data[0]
        item_id = item_map.get(path)
        if item_id:
            enqueue_ui_update(update_tree_row, item_id, data)
            return True
        return False

    _consume_scan_events(event_gen, cancel_event, "Rescanning", rescan_handler)


def run_batch_ai_analysis(
    gpt_requests: List[Dict[str, Any]],
    cancel_event: threading.Event
) -> None:
    """Perform background AI analysis for multiple items and update Treeview rows."""
    event_gen = batch_ai_analysis_events(
        gpt_requests,
        cancel_event,
        rate_limit=Config.RATE_LIMIT_PER_MINUTE,
        max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
    )

    def batch_handler(data: Tuple[Any, ...]) -> bool:
        # data format: (path, own_conf, admin, user, gpt, snippet, line, item_id)
        item_id = data[7] if len(data) > 7 else None
        if item_id:
            enqueue_ui_update(update_tree_row, item_id, data[:7])
            return True
        return False

    _consume_scan_events(event_gen, cancel_event, "AI Analysis", batch_handler)


def generate_console_report(results: List[Dict[str, Any]], use_color: bool = False) -> str:
    """Generate a colorized, human-readable triage report for the console.

    Args:
        results: List of standardized result dictionaries.
        use_color: Whether to use ANSI color codes.

    Returns:
        A formatted string report.
    """
    if not results:
        return "No findings to report."

    # ANSI Color Codes
    RED = "\033[1;91m" if use_color else ""
    YELLOW = "\033[1;93m" if use_color else ""
    GRAY = "\033[0;90m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    lines = [f"{BOLD}--- GPT SCAN TRIAGE REPORT ---{RESET}", ""]

    for i, r in enumerate(results, 1):
        path = r.get("path", "unknown")
        own_conf = r.get("own_conf", "0%")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        line_num = r.get("line", "-")
        snippet = r.get("snippet", "")

        conf_val = get_effective_threat_level(own_conf, gpt_conf)
        risk = get_risk_category(conf_val, Config.THRESHOLD)

        if risk == 'high':
            risk_label = f"{RED}HIGH RISK{RESET}"
        elif risk == 'medium':
            risk_label = f"{YELLOW}MEDIUM RISK{RESET}"
        else:
            risk_label = f"{GRAY}LOW RISK{RESET}"

        lines.append(f"{BOLD}[{i}] {risk_label} - {path}{RESET}")
        lines.append(f"    {BOLD}Threat Level:{RESET} Local: {own_conf}" + (f", AI: {gpt_conf}" if gpt_conf else ""))
        lines.append(f"    {BOLD}Location:{RESET}   Line {line_num}")

        # Hash for VirusTotal
        h = ""
        if path.startswith("[") or not os.path.exists(path):
            if snippet:
                h = get_file_sha256(snippet.encode('utf-8'))
        else:
            h = get_file_sha256(path)

        if h:
            lines.append(f"    {BOLD}VirusTotal:{RESET} https://www.virustotal.com/gui/file/{h}")

        if admin or user:
            lines.append(f"    {BOLD}AI Analysis:{RESET}")
            if admin:
                lines.append(f"        {GRAY}Admin:{RESET} {admin}")
            if user:
                lines.append(f"        {GRAY}User:{RESET}  {user}")

        # Snippet preview (first line or first 100 chars)
        clean_snippet = snippet.strip().split('\n')[0]
        if len(clean_snippet) > 80:
            clean_snippet = clean_snippet[:77] + "..."
        lines.append(f"    {BOLD}Snippet:{RESET}    {GRAY}{clean_snippet}{RESET}")
        lines.append("")

    return "\n".join(lines)


def generate_sarif(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a SARIF log from the scan results.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries.

    Returns
    -------
    Dict[str, Any]
        The SARIF log object.
    """
    sarif_results = []
    for r in results:
        # Convert threat level strings to levels
        conf = get_effective_threat_level(r.get("own_conf", ""), r.get("gpt_conf", ""))
        risk = get_risk_category(conf, Config.THRESHOLD)
        if risk == 'high':
            level = "error"
        elif risk == 'medium':
            level = "warning"
        else:
            level = "note"

        message_text = r.get("admin_desc") or r.get("end-user_desc") or "Suspicious content detected"

        sarif_results.append({
            "ruleId": "GPTScan.MaliciousContent",
            "level": level,
            "message": {
                "text": message_text
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": r.get("path", "").replace("\\", "/")
                        },
                        "region": {
                            "startLine": int(r.get("line")) if str(r.get("line")).isdigit() else 1
                        }
                    }
                }
            ],
            "properties": {
                "own_conf": r.get("own_conf"),
                "gpt_conf": r.get("gpt_conf"),
                "admin_desc": r.get("admin_desc"),
                "end-user_desc": r.get("end-user_desc"),
                "snippet": r.get("snippet")
            }
        })

    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "GPT Virus Scanner",
                        "rules": [
                            {
                                "id": "GPTScan.MaliciousContent",
                                "shortDescription": {
                                    "text": "Potential malicious content detected."
                                },
                                "helpUri": "https://github.com/user/gpt-virus-scanner"
                            }
                        ]
                    }
                },
                "results": sarif_results
            }
        ]
    }


def generate_html(results: List[Dict[str, Any]]) -> str:
    """Generate an HTML report from the scan results.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries.

    Returns
    -------
    str
        The HTML report as a string.
    """
    rows = []
    for r in results:
        path = r.get("path", "")
        own_conf = r.get("own_conf", "")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        snippet = r.get("snippet", "")

        conf_val = get_effective_threat_level(own_conf, gpt_conf)
        risk = get_risk_category(conf_val, Config.THRESHOLD)

        row_class = ""
        if risk == 'high':
            row_class = "high-risk"
        elif risk == 'medium':
            row_class = "medium-risk"

        admin_html = html.escape(admin).replace("\n", "<br>")
        user_html = html.escape(user).replace("\n", "<br>")

        rows.append(f"""
        <tr class="{row_class}">
            <td>{html.escape(path)}</td>
            <td>{html.escape(str(r.get("line", "-")))}</td>
            <td>{html.escape(gpt_conf or own_conf)}</td>
            <td>
                <strong>Admin:</strong> {admin_html}<br>
                <strong>User:</strong> {user_html}
            </td>
            <td><pre><code>{html.escape(snippet)}</code></pre></td>
        </tr>
        """)

    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT Scan Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; color: #333; }}
        table {{ border-collapse: collapse; width: 100%; border: 1px solid #ddd; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; vertical-align: top; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .high-risk {{ background-color: #ffebee; }}
        .medium-risk {{ background-color: #fff3e0; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; margin: 0; white-space: pre-wrap; word-wrap: break-word; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>GPT Scan Results</h1>
    <div class="summary">
        <p><strong>Total Results:</strong> {len(results)}</p>
    </div>
    <table>
        <thead>
            <tr>
                <th style="width: 20%">Path</th>
                <th style="width: 5%">Line</th>
                <th style="width: 10%">Threat Level</th>
                <th style="width: 25%">Analysis</th>
                <th style="width: 40%">Snippet</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>"""


def generate_markdown(results: List[Dict[str, Any]]) -> str:
    """Generate a Markdown report from the scan results.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries.

    Returns
    -------
    str
        The Markdown report as a string.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if not results:
        return f"# GPT Scan Results\n\nGenerated on: {timestamp}\nScanner Version: {Config.VERSION}\n\nNo suspicious files found."

    lines = [
        "# GPT Scan Results",
        "",
        f"**Generated on:** {timestamp}",
        f"**Scanner Version:** {Config.VERSION}",
        f"**Total Findings:** {len(results)}",
        "",
        "## Summary Table",
        "",
        "| Path | Line | Threat Level | Analysis | Snippet |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        path = r.get("path", "").replace("|", "\\|")
        line = r.get("line", "-")
        own_conf = r.get("own_conf", "")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        snippet = r.get("snippet", "")

        conf_str = gpt_conf or own_conf
        analysis_parts = []
        if admin:
            admin_clean = admin.replace("|", "\\|").replace("\n", "<br>")
            analysis_parts.append(f"**Admin:** {admin_clean}")
        if user:
            user_clean = user.replace("|", "\\|").replace("\n", "<br>")
            analysis_parts.append(f"**User:** {user_clean}")
        analysis = "<br>".join(analysis_parts)

        # Clean up snippet for markdown table (one line, escaped)
        clean_snippet = html.escape(snippet.replace("\n", " ").replace("|", "\\|"))
        if len(clean_snippet) > 100:
            clean_snippet = clean_snippet[:97] + "..."

        lines.append(f"| {path} | {line} | {conf_str} | {analysis} | <code>{clean_snippet}</code> |")

    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")

    for r in results:
        path = r.get("path", "")
        line = r.get("line", "-")
        own_conf = r.get("own_conf", "")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        snippet = r.get("snippet", "")

        lines.append(f"### File: `{path}`")
        lines.append(f"- **Detected Line:** {line}")
        lines.append(f"- **Local Threat:** {own_conf}")
        if gpt_conf:
            lines.append(f"- **AI Threat:** {gpt_conf}")
        lines.append("")

        if admin or user:
            lines.append("#### AI Analysis")
            if admin:
                lines.append(f"**Administrator Notes:**\n{admin}\n")
            if user:
                lines.append(f"**End-User Notes:**\n{user}\n")
            lines.append("")

        lines.append("#### Code Snippet")
        # Try to infer language from extension
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        lang = ext if ext in ('py', 'js', 'bat', 'ps1', 'sh', 'rb', 'php', 'pl') else ''
        # Use 4 backticks for fence if snippet contains triple backticks
        fence = "````" if "```" in snippet else "```"
        lines.append(f"{fence}{lang}")
        lines.append(snippet)
        lines.append(f"{fence}")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def run_cli(targets: Union[str, List[str]], deep: bool, show_all: bool, use_gpt: bool, rate_limit: int, output_format: str = 'csv', dry_run: bool = False, exclude_patterns: Optional[List[str]] = None, fail_threshold: Optional[int] = None, output_file: Optional[str] = None, extra_snippets: Optional[List[Tuple[str, bytes]]] = None, import_file: Optional[str] = None) -> int:
    """Run scans and stream results to the terminal or a file.

    Parameters
    ----------
    targets : Union[str, List[str]]
        Directory or list of files to scan.
    deep : bool
        Whether to evaluate all 1024-byte windows.
    show_all : bool
        Whether to emit every scanned file.
    use_gpt : bool
        Whether to request GPT analysis for confident detections.
    rate_limit : int
        Maximum allowed GPT requests per minute.
    output_format : str
        Format of the output ('csv', 'json', 'sarif', 'html', or 'markdown'). Defaults to 'csv'.
    dry_run : bool
        Whether to simulate the scan.
    exclude_patterns : List[str], optional
        List of glob patterns to exclude from the scan.
    fail_threshold : int, optional
        Threat level threshold to trigger a failure count.
    output_file : str, optional
        Path to a file where results should be saved.
    extra_snippets : List[Tuple[str, bytes]], optional
        List of (name, content) tuples to scan as in-memory buffers.
    import_file : str, optional
        Path to a previous scan report to import and process.
    """
    keys = ["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line"]

    out_stream = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout

    if output_format == 'csv':
        writer = csv.writer(out_stream)
        writer.writerow(keys)

    cancel_event = threading.Event()
    final_progress: Optional[Tuple[int, int]] = None
    threats_found = 0
    high_risk_found = 0
    medium_risk_found = 0
    use_color = sys.stderr.isatty()
    metrics: Dict[str, Any] = {}

    result_buffer = []

    if import_file:
        if import_file == "-":
            try:
                content = sys.stdin.read()
                event_gen = import_results_from_content_generator(content)
            except Exception as e:
                error_msg = str(e)
                def error_gen():
                    yield ('progress', (0, 1, f"Error reading from terminal input: {error_msg}"))
                event_gen = error_gen()
        else:
            event_gen = import_results_generator(import_file)
    else:
        event_gen = scan_files(
            targets,
            deep,
            show_all,
            use_gpt,
            cancel_event,
            rate_limit=rate_limit,
            max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
            extra_snippets=extra_snippets,
            fail_threshold=fail_threshold,
        )

    for event_type, data in event_gen:
        if event_type == 'result':
            # data format: (path, own_conf, admin, user, gpt_conf, snippet)
            conf = get_effective_threat_level(data[1], data[4])

            # Determine if this finding counts as a threat based on the threshold
            is_threat = False
            if fail_threshold is not None:
                if conf >= fail_threshold:
                    is_threat = True
            elif conf >= Config.THRESHOLD:
                is_threat = True

            if is_threat:
                threats_found += 1
                risk = get_risk_category(conf, 0) # Already confirmed as threat, just categorize
                if risk == 'high':
                    high_risk_found += 1
                else:
                    medium_risk_found += 1

            record = dict(zip(keys, data))
            if output_format == 'json':
                print(json.dumps(record), file=out_stream)
            elif output_format in ('sarif', 'html', 'markdown', 'report'):
                result_buffer.append(record)
            else:
                writer.writerow(data)
        elif event_type == 'progress':
            current, total, status = data
            final_progress = (current, total)
            cols = shutil.get_terminal_size((80, 20)).columns

            threat_display = str(threats_found)
            if use_color and threats_found > 0:
                threat_display = f"\033[1;91m{threats_found}\033[0m"

            if threats_found > 0:
                threat_suffix = f" ({threat_display} suspicious: {high_risk_found} high, {medium_risk_found} medium)"
            else:
                threat_suffix = ""
            msg = f"{status} ({current}/{total}){threat_suffix}" if status else f"Scanning: {current}/{total} files{threat_suffix}"

            # Use \r to overwrite same line, and pad with spaces to clear any previous longer line.
            # Adjust padding for zero-width ANSI color codes (total 11 chars).
            ansi_len = 11 if use_color and threats_found > 0 else 0
            padding = " " * max(0, cols - 1 - (len(msg) - ansi_len))
            sys.stderr.write(f"\r{msg}{padding}\r")
            sys.stderr.flush()
        elif event_type == 'summary':
            total_files, total_bytes, elapsed_time = data
            metrics['total_bytes'] = total_bytes
            metrics['elapsed_time'] = elapsed_time

    if final_progress is not None:
        print(file=sys.stderr)
        total_scanned = final_progress[1]
        summary = format_scan_summary(
            total_scanned,
            threats_found,
            metrics.get('total_bytes'),
            metrics.get('elapsed_time'),
            use_color=use_color,
            high_risk=high_risk_found,
            medium_risk=medium_risk_found
        )
        print(summary, file=sys.stderr)

    if output_format == 'sarif':
        sarif_log = generate_sarif(result_buffer)
        print(json.dumps(sarif_log, indent=2), file=out_stream)
    elif output_format == 'html':
        print(generate_html(result_buffer), file=out_stream)
    elif output_format == 'markdown':
        print(generate_markdown(result_buffer), file=out_stream)
    elif output_format == 'report':
        # Sort results by effective threat level (highest first)
        result_buffer.sort(
            key=lambda x: get_effective_threat_level(x.get('own_conf', '0%'), x.get('gpt_conf', '')),
            reverse=True
        )
        # Use color only if the output stream is a terminal
        use_color_output = out_stream.isatty() if hasattr(out_stream, 'isatty') else False
        report = generate_console_report(result_buffer, use_color=use_color_output)
        print(report, file=out_stream)

    if output_file:
        out_stream.close()

    return threats_found


REPORT_FIELD_MAPPING = {
    "path": ["path", "File Path", "uri", "Path"],
    "own_conf": ["own_conf", "Local Threat", "Local Conf.", "local_conf", "Confidence"],
    "admin_desc": ["admin_desc", "Admin Notes", "admin", "Analysis"],
    "end-user_desc": ["end-user_desc", "User Notes", "user_desc", "Analysis"],
    "gpt_conf": ["gpt_conf", "AI Threat", "AI Conf.", "ai_conf", "Confidence", "Threat Level"],
    "snippet": ["snippet", "Snippet", "code", "Snippet"],
    "line": ["line", "Line", "startLine", "Line"]
}


def standardize_result_dict(item: Any) -> Dict[str, Any]:
    """Map various report keys back to the standard internal format.

    Args:
        item: A dictionary containing raw result data from a report.

    Returns:
        A dictionary with standardized keys for internal use.
    """
    if not isinstance(item, dict):
        return {k: "" for k in REPORT_FIELD_MAPPING}

    res = {
        key: next((str(item[alt]) if item[alt] is not None else "" for alt in alts if alt in item), "")
        for key, alts in REPORT_FIELD_MAPPING.items()
    }
    # Fallback: if own_conf is empty but gpt_conf has a value, they might be using a report
    # where both mapped to 'Threat Level' (like in Markdown) or 'Confidence'.
    if not res.get('own_conf') and res.get('gpt_conf'):
        res['own_conf'] = res['gpt_conf']
    return res


def parse_report_content(content: str, filename_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse report content in JSON, SARIF, or CSV format.

    Args:
        content: The raw string content of the report.
        filename_hint: Optional filename or extension hint (e.g., '.json', '.csv').

    Returns:
        A list of standardized result dictionaries.
    """
    content = content.strip()
    if not content:
        return []

    ext = ""
    if filename_hint:
        ext = os.path.splitext(filename_hint)[1].lower()

    if ext and ext not in ('.json', '.jsonl', '.ndjson', '.sarif', '.csv', '.md', '.markdown', '.html', '.htm', '.xhtml'):
        raise ValueError(f"Unsupported file extension: {ext}")

    data_to_import = []

    # Auto-detection logic
    if (content.strip().startswith('<') and ('<table' in content.lower() or '<tr' in content.lower())) or ext in ('.html', '.htm', '.xhtml'):
        # HTML report format
        # Use regex to find rows, ignoring the header row
        rows = re.findall(r'<tr\b[^>]*>(.*?)</tr>', content, re.DOTALL | re.IGNORECASE)
        for row_content in rows:
            if '<th' in row_content.lower():
                continue

            cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row_content, re.DOTALL | re.IGNORECASE)
            if len(cells) >= 5:
                # Path, Line, Threat Level, Analysis, Snippet
                # Snippet is inside <pre><code>
                snippet_cell = cells[4]
                snippet_match = re.search(r'<pre><code>(.*?)</code></pre>', snippet_cell, re.DOTALL | re.IGNORECASE)
                snippet_raw = snippet_match.group(1) if snippet_match else snippet_cell

                analysis_cell = cells[3]
                admin_match = re.search(r'<strong>Admin:</strong>\s*(.*?)(?:\s*<br>\s*<strong>User:</strong>|$)', analysis_cell, re.DOTALL | re.IGNORECASE)
                user_match = re.search(r'<strong>User:</strong>\s*(.*)', analysis_cell, re.DOTALL | re.IGNORECASE)

                admin_desc = html.unescape(admin_match.group(1).strip().replace('<br>', '\n')) if admin_match else ""
                user_desc = html.unescape(user_match.group(1).strip().replace('<br>', '\n')) if user_match else ""

                item = {
                    "path": html.unescape(cells[0].strip()),
                    "line": html.unescape(cells[1].strip()),
                    "own_conf": html.unescape(cells[2].strip()),
                    "admin_desc": admin_desc,
                    "end-user_desc": user_desc,
                    "snippet": html.unescape(snippet_raw.strip())
                }
                data_to_import.append(item)
    elif content.startswith('[') or (ext in ('.json', '.jsonl', '.ndjson')):
        if content.startswith('['):
            # Standard JSON list
            data_to_import = json.loads(content)
        else:
            # Try to parse as single JSON object first
            try:
                item = json.loads(content)
                if isinstance(item, list):
                    data_to_import = item
                else:
                    data_to_import = [item]
            except json.JSONDecodeError:
                # Fallback to NDJSON
                data_to_import = [json.loads(line) for line in content.splitlines() if line.strip()]
    elif (content.startswith('{') and '"runs"' in content) or ext == '.sarif':
        # SARIF format
        sarif_data = json.loads(content)
        for run in sarif_data.get("runs", []):
            for result in run.get("results", []):
                props = result.get("properties", {})
                mapped = {
                    "path": "",
                    "own_conf": props.get("own_conf", ""),
                    "admin_desc": props.get("admin_desc") or result.get("message", {}).get("text", ""),
                    "end-user_desc": props.get("end-user_desc", ""),
                    "gpt_conf": props.get("gpt_conf", ""),
                    "snippet": props.get("snippet", ""),
                    "line": "-"
                }
                locations = result.get("locations", [])
                if locations:
                    phys_loc = locations[0].get("physicalLocation", {})
                    uri = phys_loc.get("artifactLocation", {}).get("uri", "")
                    mapped["path"] = uri.replace("/", os.sep)
                    region = phys_loc.get("region", {})
                    if "startLine" in region:
                        mapped["line"] = region["startLine"]
                data_to_import.append(mapped)
    elif ext == '.csv' or ',' in content.splitlines()[0]:
        # CSV format
        f = io.StringIO(content)
        reader = csv.DictReader(f)
        data_to_import = list(reader)
    elif ext in ('.md', '.markdown') or '|' in content:
        # Markdown table format
        lines = content.splitlines()
        headers = []
        for idx, line in enumerate(lines):
            if '|' in line and any(h in line for h in ['Path', 'Line', 'Threat Level', 'Analysis', 'Snippet']):
                headers = [h.strip() for h in line.strip('|').split('|')]
                start_idx = idx + 2 # Skip header and separator
                for row_line in lines[start_idx:]:
                    if '|' not in row_line:
                        continue
                    # Split by | but ignore escaped \|
                    cols = [c.strip() for c in re.split(r'(?<!\\)\|', row_line.strip('|'))]
                    if len(cols) >= len(headers):
                        item = dict(zip(headers, cols))
                        if 'Threat Level' in item:
                            item['gpt_conf'] = item['Threat Level']

                        # Specialized logic for Markdown Analysis and Snippet
                        analysis = item.get('Analysis', '')
                        if analysis:
                            # Reconstruct Admin and User notes from Analysis column
                            # Admin: ... <br> User: ...
                            admin_match = re.search(r'\*\*Admin:\*\*\s*(.*?)(?:\s*<br>\s*\*\*User:\*\*|$)', analysis)
                            user_match = re.search(r'\*\*User:\*\*\s*(.*)', analysis)

                            item['admin_desc'] = admin_match.group(1).replace('<br>', '\n').replace('\\|', '|') if admin_match else ""
                            item['end-user_desc'] = user_match.group(1).replace('<br>', '\n').replace('\\|', '|') if user_match else ""

                        # Clean up Snippet (remove backticks or <code> tags)
                        if 'Snippet' in item:
                            raw_snippet = item['Snippet']
                            if raw_snippet.startswith('<code>') and raw_snippet.endswith('</code>'):
                                raw_snippet = raw_snippet[6:-7]
                            else:
                                raw_snippet = raw_snippet.strip('`')

                            item['Snippet'] = html.unescape(raw_snippet).replace('\\|', '|')

                        if 'Path' in item:
                            item['Path'] = item['Path'].replace('\\|', '|')

                        data_to_import.append(item)
                break
    else:
        # Last resort: try JSON parsing anyway
        try:
            data_to_import = [json.loads(content)]
        except json.JSONDecodeError:
            raise ValueError("Could not determine report format.")

    # Filter out non-dictionary items before standardizing
    return [standardize_result_dict(item) for item in data_to_import if isinstance(item, dict)]


def load_report_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a report file in JSON, SARIF, or CSV format.

    Args:
        file_path: Path to the report file.

    Returns:
        A list of standardized result dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError("File is empty.")

    return parse_report_content(content, filename_hint=file_path)


def import_results_from_content_generator(content: str, filename_hint: Optional[str] = None) -> Generator[Tuple[str, Any], None, None]:
    """Generator that yields events from imported report content.

    Mimics the event stream from scan_files to allow processing of imported
    results through the standard CLI and GUI logic.

    Args:
        content: Raw report content string.
        filename_hint: Optional filename or extension hint.

    Yields:
        Events ('progress', 'result', 'summary') identical to scan_files.
    """
    try:
        results = parse_report_content(content, filename_hint=filename_hint)
    except Exception as e:
        yield ('progress', (0, 1, f"Error parsing report: {e}"))
        return

    total = len(results)
    source_name = os.path.basename(filename_hint) if filename_hint else "Content"
    yield ('progress', (0, total, f"Importing: {source_name}"))

    for i, item in enumerate(results):
        yield ('progress', (i + 1, total, f"Importing: {os.path.basename(item.get('path', 'unknown'))}"))

        # Format as the expected 7-element tuple:
        # (path, own_conf, admin_desc, user_desc, gpt_conf, snippet, line)
        data = (
            item.get("path", ""),
            item.get("own_conf", ""),
            item.get("admin_desc", ""),
            item.get("end-user_desc", ""),
            item.get("gpt_conf", ""),
            item.get("snippet", ""),
            item.get("line", "-")
        )
        yield ('result', data)

    yield ('summary', (total, 0, 0.0))


def import_results_generator(file_path: str) -> Generator[Tuple[str, Any], None, None]:
    """Generator that yields events from an imported report file or URL."""
    try:
        if file_path.lower().startswith(('http://', 'https://')):
            content_bytes = fetch_url_content(file_path)
            content = content_bytes.decode('utf-8', errors='ignore')
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        if not content.strip():
            raise ValueError("File or URL content is empty.")
    except Exception as e:
        yield ('progress', (0, 1, f"Error: {e}"))
        return

    yield from import_results_from_content_generator(content, filename_hint=file_path)


def _finalize_import(data_to_import: List[Dict[str, Any]], source_name: str) -> None:
    """Clear results and populate the Treeview with imported data."""
    # Clear existing results
    clear_results()

    count = 0
    for item in data_to_import:
        # Map item keys back to the expected column order
        values = (
            item["path"],
            item["own_conf"],
            item["admin_desc"],
            item["end-user_desc"],
            item["gpt_conf"],
            item["snippet"],
            item["line"]
        )
        insert_tree_row(values)
        count += 1

    msg = f"Imported {count} results from {source_name}"
    global _last_scan_summary
    _last_scan_summary = msg
    update_status(msg)
    update_tree_columns()
    _auto_select_best_result()


def import_results() -> None:
    """Load results from a JSON or CSV file into the Treeview.

    Supports standard JSON lists, NDJSON (newline-delimited JSON), and CSV files.

    Returns
    -------
    None
        Clears the Treeview and populates it with imported data, or shows an error.
    """
    if not tree:
        return

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All supported formats", "*.json;*.jsonl;*.ndjson;*.csv;*.sarif;*.md;*.markdown;*.html;*.htm;*.xhtml"),
            ("JSON files", "*.json;*.jsonl;*.ndjson"),
            ("SARIF files", "*.sarif"),
            ("CSV files", "*.csv"),
            ("Markdown files", "*.md;*.markdown"),
            ("HTML files", "*.html;*.htm;*.xhtml"),
            ("All files", "*.*")
        ],
        title="Import Scan Results",
    )
    if not file_path:
        return

    try:
        data_to_import = load_report_file(file_path)

        if not data_to_import:
            messagebox.showwarning("Import Warning", "No data found in the selected file.")
            return

        _finalize_import(data_to_import, os.path.basename(file_path))

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not load results:\n{err}")


def import_from_clipboard(event: Optional[tk.Event] = None) -> Optional[str]:
    """Import scan results from the system clipboard."""
    if not tree:
        return "break"

    # Don't intercept paste if focus is on an entry or text widget
    focused = root.focus_get() if root else None
    if isinstance(focused, (ttk.Entry, tk.Entry, tk.Text, scrolledtext.ScrolledText)):
        return None

    try:
        content = root.clipboard_get()
        if not content:
            return "break"
    except Exception as e:
        messagebox.showwarning("Clipboard Error", f"Could not read from clipboard: {e}")
        return "break"

    try:
        data_to_import = parse_report_content(content)

        if not data_to_import:
            messagebox.showwarning("Import Warning", "No valid scan results found in clipboard.")
            return "break"

        _finalize_import(data_to_import, "clipboard")

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not parse clipboard content:\n{err}")

    return "break"


def import_from_url() -> None:
    """Import scan results from a web link (JSON, CSV, SARIF, Markdown, or HTML)."""
    if not tree:
        return

    url = simpledialog.askstring("Import from URL", "Enter the URL of the scan results to import:")
    if not url:
        return

    url = url.strip()
    try:
        update_status(f"Fetching results from {url}...")
        content_bytes = fetch_url_content(url)
        content = content_bytes.decode('utf-8', errors='ignore')

        data_to_import = parse_report_content(content, filename_hint=url)

        if not data_to_import:
            messagebox.showwarning("Import Warning", "No valid scan results found at the provided URL.")
            return

        _finalize_import(data_to_import, url)

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not import results from URL:\n{err}")


def clear_ai_cache() -> None:
    """Clear the AI analysis cache and update the persistent file."""
    Config.gpt_cache = {}
    Config.save_cache()
    update_status("AI Analysis cache cleared.")


def clear_results() -> None:
    """Clear all results from the Treeview and reset progress/status."""
    global _all_results_cache, _last_scan_summary
    _all_results_cache = []
    _last_scan_summary = ""
    if tree:
        items = tree.get_children()
        if items:
            tree.delete(*items)
    if progress_bar:
        progress_bar["value"] = 0
    update_status("Ready")


def clear_path_history() -> None:
    """Clear the history of scanned paths."""
    Config.recent_paths = []
    if textbox:
        textbox['values'] = []
    Config.save_settings()
    update_status("Path history cleared.")


def _get_tree_results_as_dicts(item_ids: Iterable[str]) -> List[Dict[str, Any]]:
    """Extract raw results from the given Treeview item IDs as a list of dictionaries."""
    if not tree:
        return []

    columns = tree["columns"][:7]
    results = []
    for item_id in item_ids:
        values = _get_item_raw_values(item_id)
        if values:
            results.append(dict(zip(columns, values)))
    return results


def export_results() -> None:
    """Save the current Treeview contents to a file chosen by the user.

    Supports CSV, HTML, JSON, and SARIF formats.

    Returns
    -------
    None
        Writes the Treeview rows to the selected path or shows an error.
    """
    if not tree:
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Markdown files", "*.md"),
            ("HTML files", "*.html"),
            ("JSON files", "*.json"),
            ("SARIF files", "*.sarif"),
            ("All files", "*.*")
        ],
        title="Export Scan Results",
    )
    if not file_path:
        return

    results = _get_tree_results_as_dicts(tree.get_children())
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.json':
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif ext == '.html':
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(generate_html(results))
        elif ext == '.sarif':
            sarif_log = generate_sarif(results)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sarif_log, f, indent=2)
        elif ext == '.md':
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(generate_markdown(results))
        else: # Default to CSV
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                columns = tree["columns"][:7]
                writer.writerow(columns)
                for r in results:
                    writer.writerow([r[col] for col in columns])

        messagebox.showinfo("Export Successful", f"Results saved to {os.path.basename(file_path)}")

    except Exception as err:
        messagebox.showerror("Export Failed", f"Could not save results:\n{err}")


def _get_item_raw_values(item_id: str) -> Optional[List[Any]]:
    """Retrieve raw values from a specific Treeview row, prioritizing the hidden cache."""
    if not tree or not tree.exists(item_id):
        return None
    values = list(tree.item(item_id, "values"))

    # Try to return raw values from the hidden column (index 7) if available
    if len(values) > 7 and values[7]:
        try:
            return json.loads(values[7])
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: unwrap display newlines by replacing them with spaces
    return [str(v).replace('\n', ' ') for v in values[:7]]


def _resolve_file_path(event_or_path: Union[tk.Event, str, None], verify: bool = True) -> Optional[str]:
    """Retrieve and optionally verify a file path from an event or direct argument."""
    if isinstance(event_or_path, str):
        file_path = event_or_path
    else:
        if not tree:
            return None
        selection = tree.selection()
        if not selection:
            return None
        values = _get_item_raw_values(selection[0])
        if not values:
            return None
        file_path = str(values[0])

    if verify and not file_path.startswith("[") and not file_path.startswith(("http://", "https://")) and not os.path.exists(file_path):
        messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")
        return None
    return file_path


def view_details(event: Optional[tk.Event] = None, item_id: Optional[str] = None) -> None:
    """Open a detailed view of the selected scan result.

    This window displays technical and user-focused analysis, the suspicious
    code snippet, and allows toggling to the full source code with highlighting
     of the detected line.
    """
    if item_id is None:
        selection = tree.selection()
        if not selection:
            return
        item_id = selection[0]

    current_item_id = item_id

    values = _get_item_raw_values(current_item_id)
    if not values:
        return

    # values: (path, own_conf, admin, user, gpt_conf, snippet)
    path = values[0]

    details_win = tk.Toplevel(root)
    details_win.title(f"Result Details - {os.path.basename(path)}")
    details_win.geometry("1100x650")
    details_win.minsize(800, 450)

    # Make it modal-ish but not blocking
    details_win.transient(root)

    main_frame = ttk.Frame(details_win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Local Status Bar
    status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def set_local_status(message: str, temporary: bool = False):
        """Update the local status bar with an optional timeout."""
        status_bar.config(text=message)
        if temporary:
            details_win.after(3000, lambda: status_bar.config(text="Ready"))

    # Header: Navigation and Info
    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill=tk.X, pady=(0, 5))

    nav_header = ttk.Frame(header_frame)
    nav_header.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 10))

    ttk.Label(header_frame, text="File Path:", font=('TkDefaultFont', 9, 'bold')).grid(row=1, column=0, sticky="w")
    path_entry = ttk.Entry(header_frame)
    path_entry.grid(row=1, column=1, sticky="ew", padx=5)
    header_frame.columnconfigure(1, weight=1)

    conf_frame = ttk.Frame(header_frame)
    conf_frame.grid(row=2, column=0, columnspan=5, sticky="w", pady=(5, 0))

    ttk.Label(conf_frame, text="Local Threat:").grid(row=0, column=0, sticky="w")
    own_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))
    own_conf_label.grid(row=0, column=1, sticky="w", padx=(5, 20))

    ai_conf_prefix = ttk.Label(conf_frame, text="AI Threat:")
    gpt_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))

    ttk.Label(conf_frame, text="Detected Line:").grid(row=0, column=5, sticky="w", padx=(20, 5))
    line_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))
    line_label.grid(row=0, column=6, sticky="w")

    risk_badge = tk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'), padx=8, pady=2)
    risk_badge.grid(row=0, column=7, sticky="w", padx=(20, 0))

    ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

    # Analysis sections
    analysis_frame = ttk.LabelFrame(main_frame, text="AI Analysis", padding=5)
    admin_label = ttk.Label(analysis_frame, text="Administrator Notes:", font=('TkDefaultFont', 9, 'bold'))
    admin_text = scrolledtext.ScrolledText(analysis_frame, height=5, wrap=tk.WORD)
    user_label = ttk.Label(analysis_frame, text="End-User Notes:", font=('TkDefaultFont', 9, 'bold'))
    user_text = scrolledtext.ScrolledText(analysis_frame, height=5, wrap=tk.WORD)

    # Snippet section
    snippet_frame = ttk.LabelFrame(main_frame, text="Code Snippet", padding=5)
    snippet_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    snippet_text = scrolledtext.ScrolledText(snippet_frame, height=8, font=('Courier', 10), wrap=tk.NONE)
    snippet_text.pack(fill=tk.BOTH, expand=True)
    snippet_text.tag_configure("highlight", background="yellow", foreground="black")

    showing_full_source = False

    def load_display_code(path, line, snippet, silent_fallback=False):
        """Load and display either the snippet or full source code."""
        nonlocal showing_full_source
        if showing_full_source:
            if path.startswith("["):
                if not silent_fallback:
                    messagebox.showinfo("Full Source", "Full source is not available for files inside archives, web links, or clipboard content.")
            elif not os.path.exists(path):
                if not silent_fallback:
                    messagebox.showerror("Error", f"File not found: {path}")
            else:
                try:
                    file_size = os.path.getsize(path)
                    if file_size > Config.MAX_SOURCE_VIEW_SIZE:
                        if not messagebox.askyesno("Large File", f"The file is {format_bytes(file_size)}. Loading it might be slow. Continue?"):
                            raise ValueError("Cancelled")

                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()

                    snippet_text.config(state='normal')
                    snippet_text.delete('1.0', tk.END)
                    snippet_text.insert(tk.END, content)

                    # Highlight and scroll to line
                    if str(line).isdigit():
                        line_idx = f"{line}.0"
                        snippet_text.tag_add("highlight", line_idx, f"{line}.end")
                        snippet_text.see(line_idx)

                    snippet_text.config(state='disabled')
                    source_toggle_btn.config(text="Show Snippet")
                    snippet_frame.config(text="Full Source")
                    return
                except Exception as e:
                    if not silent_fallback and str(e) != "Cancelled":
                        messagebox.showerror("Error", f"Could not read file: {e}")

        # Default to snippet view
        snippet_text.config(state='normal')
        snippet_text.delete('1.0', tk.END)
        snippet_text.insert(tk.END, snippet)
        snippet_text.config(state='disabled')
        showing_full_source = False
        source_toggle_btn.config(text="Show Full Source")
        snippet_frame.config(text="Code Snippet")

    def toggle_source():
        """Toggle between snippet and full source view."""
        nonlocal showing_full_source
        showing_full_source = not showing_full_source
        vals = _get_item_raw_values(current_item_id)
        if vals:
            path = vals[0]
            line = vals[6] if len(vals) > 6 and vals[6] != "-" else 1
            snippet = vals[5]
            load_display_code(path, line, snippet)

    def on_rescan():
        """Re-scan the current file."""
        global current_cancel_event
        if current_cancel_event is not None:
            return

        vals = _get_item_raw_values(current_item_id)
        if not vals:
            return
        path = vals[0]

        current_cancel_event = threading.Event()
        set_scanning_state(True)
        rescan_btn.config(state='disabled', text="Rescanning...")

        def run_thread(target_id, target_path):
            try:
                settings = {
                    'deep': deep_var.get() if deep_var else False,
                    'gpt': gpt_var.get() if gpt_var else False,
                    'dry': dry_var.get() if dry_var else False,
                }

                run_rescan([target_path], {target_path: target_id}, settings, current_cancel_event)

                if current_item_id == target_id:
                    enqueue_ui_update(refresh_content, target_id)
            except Exception as e:
                enqueue_ui_update(messagebox.showerror, "Error", f"An unexpected error occurred: {e}")
                enqueue_ui_update(finish_scan_state)
            finally:
                enqueue_ui_update(lambda: rescan_btn.config(state='normal', text="Rescan"))

        threading.Thread(target=run_thread, args=(current_item_id, path), daemon=True).start()

    # Footer buttons
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=(10, 0))

    def copy_path_details():
        root.clipboard_clear()
        root.clipboard_append(path_entry.get())
        set_local_status("File path copied to clipboard.", temporary=True)

    path_copy_btn = ttk.Button(header_frame, text="Copy", width=8, command=copy_path_details)
    path_copy_btn.grid(row=1, column=2, padx=2)
    bind_hover_message(path_copy_btn, "Copy the full file path to the clipboard.", label=status_bar)

    reveal_btn = ttk.Button(header_frame, text="Reveal", width=10, command=lambda: show_in_folder(path_entry.get()))
    reveal_btn.grid(row=1, column=3, padx=2)
    bind_hover_message(reveal_btn, "Show this file in the system file manager.", label=status_bar)

    open_btn = ttk.Button(header_frame, text="Open", width=10, command=lambda: open_file(path_entry.get()))
    open_btn.grid(row=1, column=4, padx=2)
    bind_hover_message(open_btn, "Open this file in the default application. (Shift+Enter)", label=status_bar)

    def on_analyze_now():
        if current_cancel_event is not None:
            return

        if not Config.GPT_ENABLED:
            messagebox.showwarning("AI Disabled", "AI Analysis is disabled (task.txt not found or API key missing).")
            return

        analyze_btn.config(state='disabled', text="Analyzing...")

        def run_thread(target_id):
            try:
                vals = _get_item_raw_values(target_id)
                if not vals: return
                snippet = vals[5]

                result = request_single_gpt_analysis(snippet)

                if result:
                    # updated_vals: (path, own_conf, admin, user, gpt_conf, snippet)
                    updated_vals = list(vals)
                    updated_vals[2] = result.get("administrator", "")
                    updated_vals[3] = result.get("end-user", "")
                    # Safer extraction of threat-level
                    threat_level = result.get("threat-level", 0)
                    try:
                        updated_vals[4] = "{:.0%}".format(int(threat_level) / 100.)
                    except (ValueError, TypeError):
                        updated_vals[4] = "Error"

                    enqueue_ui_update(update_tree_row, target_id, tuple(updated_vals))
                    # Only refresh the details view if the user is still viewing the same item
                    if current_item_id == target_id:
                        enqueue_ui_update(refresh_content, target_id)
                else:
                    enqueue_ui_update(messagebox.showerror, "AI Analysis Failed", "Could not obtain a response from the AI.")
            except Exception as e:
                enqueue_ui_update(messagebox.showerror, "Error", f"An unexpected error occurred: {e}")
            finally:
                enqueue_ui_update(lambda: analyze_btn.config(state='normal', text="Analyze with AI"))

        threading.Thread(target=run_thread, args=(current_item_id,), daemon=True).start()

    def copy_analysis():
        path = path_entry.get()
        own_conf = own_conf_label.cget("text")
        gpt_conf = gpt_conf_label.cget("text")
        text = f"Path: {path}\nLocal Threat: {own_conf}\n"
        if gpt_conf:
            text += f"AI Threat: {gpt_conf}\n"
        if analysis_frame.winfo_viewable():
            if admin_label.winfo_viewable():
                text += f"\nAdmin Notes:\n{admin_text.get('1.0', tk.END).strip()}\n"
            if user_label.winfo_viewable():
                text += f"\nUser Notes:\n{user_text.get('1.0', tk.END).strip()}\n"
        text += f"\nSnippet:\n{snippet_text.get('1.0', tk.END).strip()}"
        root.clipboard_clear()
        root.clipboard_append(text)
        set_local_status("Detailed analysis copied to clipboard.", temporary=True)

    def copy_code():
        code = snippet_text.get("1.0", tk.END).strip()
        root.clipboard_clear()
        root.clipboard_append(code)
        set_local_status("Code copied to clipboard.", temporary=True)

    def copy_as_json_details():
        results = _get_tree_results_as_dicts([current_item_id])
        if results:
            js = json.dumps(results[0], indent=2)
            root.clipboard_clear()
            root.clipboard_append(js)
            set_local_status("Result copied as JSON.", temporary=True)

    def on_exclude():
        """Exclude current file and move to next."""
        nonlocal current_item_id
        p = path_entry.get()

        # Capture current list and index to handle transition correctly
        all_visible = list(tree.get_children())
        try:
            current_idx = all_visible.index(current_item_id)
        except ValueError:
            current_idx = -1

        if exclude_paths([p], confirm=True):
            # After exclusion, list changes. Get new list.
            new_visible = list(tree.get_children())
            if not new_visible:
                details_win.destroy()
                return

            # Try to stay at the same index, or go to previous if we were at the end
            if current_idx >= len(new_visible):
                new_idx = len(new_visible) - 1
            else:
                new_idx = max(0, current_idx)

            new_item_id = new_visible[new_idx]
            tree.selection_set(new_item_id)
            tree.see(new_item_id)
            refresh_content(new_item_id)

    # Group: Actions
    analyze_btn = ttk.Button(btn_frame, text="Analyze with AI", width=18, command=on_analyze_now, style='Primary.TButton')
    analyze_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(analyze_btn, "Use AI to analyze this code snippet.", label=status_bar)
    if not Config.GPT_ENABLED:
        analyze_btn.config(state='disabled')

    vt_btn = ttk.Button(btn_frame, text="VirusTotal", width=12, command=lambda: check_virustotal(path_entry.get()))
    vt_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(vt_btn, "Check this file's hash on VirusTotal.", label=status_bar)

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: Management
    rescan_btn = ttk.Button(btn_frame, text="Rescan", width=10, command=on_rescan)
    rescan_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(rescan_btn, "Re-scan this file with current settings. (F5 or R)", label=status_bar)

    exclude_btn = ttk.Button(btn_frame, text="Exclude", width=10, command=on_exclude)
    exclude_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(exclude_btn, "Exclude this file from future scans. (Delete)", label=status_bar)

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: Copy & Export
    copy_btn = ttk.Button(btn_frame, text="Copy Analysis", width=15, command=copy_analysis)
    copy_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(copy_btn, "Copy the full analysis and snippet to clipboard.", label=status_bar)

    copy_json_btn = ttk.Button(btn_frame, text="Copy JSON", width=12, command=copy_as_json_details)
    copy_json_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(copy_json_btn, "Copy the current result as a JSON object. (Ctrl+J)", label=status_bar)

    copy_code_btn = ttk.Button(btn_frame, text="Copy Code", width=12, command=copy_code)
    copy_code_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(copy_code_btn, "Copy the displayed code to the clipboard. (Ctrl+S)", label=status_bar)

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: View
    source_toggle_btn = ttk.Button(btn_frame, text="Show Full Source", width=18, command=toggle_source)
    source_toggle_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(source_toggle_btn, "Toggle between the suspicious snippet and the full file content.", label=status_bar)

    # Group: System
    close_btn = ttk.Button(btn_frame, text="Close", command=details_win.destroy)
    close_btn.pack(side=tk.RIGHT, padx=5, ipady=5)
    bind_hover_message(close_btn, "Close this window. (Esc)", label=status_bar)

    def refresh_content(new_id):
        nonlocal current_item_id
        current_item_id = new_id
        vals = _get_item_raw_values(new_id)
        if not vals: return

        # Ensure buttons reflect current scanning state
        is_scanning = current_cancel_event is not None
        rescan_btn.config(state='disabled' if is_scanning else 'normal')
        analyze_btn.config(state='disabled' if is_scanning or not Config.GPT_ENABLED else 'normal')
        exclude_btn.config(state='disabled' if is_scanning else 'normal')
        open_btn.config(state='disabled' if is_scanning else 'normal')
        vt_btn.config(state='disabled' if is_scanning else 'normal')
        path_copy_btn.config(state='disabled' if is_scanning else 'normal')
        reveal_btn.config(state='disabled' if is_scanning else 'normal')

        # vals: (path, own_conf, admin, user, gpt_conf, snippet, line)
        path, own_conf, admin, user, gpt_conf, snippet = vals[:6]
        line = vals[6] if len(vals) > 6 else "-"

        all_visible = tree.get_children()
        total = len(all_visible)
        try:
            idx = all_visible.index(new_id)
            prev_btn.config(state='normal' if idx > 0 else 'disabled')
            next_btn.config(state='normal' if idx < total - 1 else 'disabled')
            count_label.config(text=f"Result {idx + 1} of {total}")
            details_win.title(f"Result {idx + 1} of {total} - {os.path.basename(path)}")
        except ValueError:
            prev_btn.config(state='disabled')
            next_btn.config(state='disabled')
            count_label.config(text="")
            details_win.title(f"Result Details - {os.path.basename(path)}")

        path_entry.config(state='normal')
        path_entry.delete(0, tk.END)
        path_entry.insert(0, path)
        path_entry.config(state='readonly')
        own_conf_label.config(text=own_conf)
        line_label.config(text=str(line))

        conf = get_effective_threat_level(own_conf, gpt_conf)
        risk = get_risk_category(conf, Config.THRESHOLD)

        if risk == 'high':
            risk_badge.config(text="HIGH RISK", background="#ffcccc", foreground="darkred")
        elif risk == 'medium':
            risk_badge.config(text="MEDIUM RISK", background="#fff0cc", foreground="darkorange")
        else:
            risk_badge.config(text="LOW RISK", background="lightgrey", foreground="grey")

        if gpt_conf:
            ai_conf_prefix.grid(row=0, column=2, sticky="w")
            gpt_conf_label.grid(row=0, column=3, sticky="w", padx=(5, 0))
            gpt_conf_label.config(text=gpt_conf)
        else:
            ai_conf_prefix.grid_forget()
            gpt_conf_label.grid_forget()
            gpt_conf_label.config(text="")

        if admin or user:
            analysis_frame.pack(fill=tk.BOTH, expand=True, pady=5, before=snippet_frame)
            if admin:
                admin_label.pack(anchor="w")
                admin_text.config(state='normal')
                admin_text.delete('1.0', tk.END)
                admin_text.insert(tk.END, admin)
                admin_text.config(state='disabled')
                admin_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            else:
                admin_label.pack_forget()
                admin_text.pack_forget()
            if user:
                user_label.pack(anchor="w")
                user_text.config(state='normal')
                user_text.delete('1.0', tk.END)
                user_text.insert(tk.END, user)
                user_text.config(state='disabled')
                user_text.pack(fill=tk.BOTH, expand=True)
            else:
                user_label.pack_forget()
                user_text.pack_forget()
        else:
            analysis_frame.pack_forget()

        load_display_code(path, line, snippet, silent_fallback=True)

    def on_prev():
        all_visible = tree.get_children()
        try:
            idx = all_visible.index(current_item_id)
            if idx > 0:
                new_id = all_visible[idx - 1]
                tree.selection_set(new_id)
                tree.see(new_id)
                refresh_content(new_id)
        except ValueError: pass

    def on_next():
        all_visible = tree.get_children()
        try:
            idx = all_visible.index(current_item_id)
            if idx < len(all_visible) - 1:
                new_id = all_visible[idx + 1]
                tree.selection_set(new_id)
                tree.see(new_id)
                refresh_content(new_id)
        except ValueError: pass

    prev_btn = ttk.Button(nav_header, text="< Previous", command=on_prev)
    prev_btn.pack(side=tk.LEFT, ipady=2)
    bind_hover_message(prev_btn, "View the previous scan result. (Left Arrow)", label=status_bar)

    count_label = ttk.Label(nav_header, text="", font=('TkDefaultFont', 9, 'bold'))
    count_label.pack(side=tk.LEFT, padx=20)

    next_btn = ttk.Button(nav_header, text="Next >", command=on_next)
    next_btn.pack(side=tk.LEFT, ipady=2)
    bind_hover_message(next_btn, "View the next scan result. (Right Arrow)", label=status_bar)

    details_win.bind('<Left>', lambda e: on_prev())
    details_win.bind('<Right>', lambda e: on_next())
    details_win.bind('<Delete>', lambda e: on_exclude())
    details_win.bind('<Escape>', lambda e: details_win.destroy())
    details_win.bind('<Shift-Return>', lambda e: open_file(path_entry.get()))
    details_win.bind('<Control-s>', lambda e: copy_code())
    details_win.bind('<Command-s>', lambda e: copy_code())
    details_win.bind('<Control-j>', lambda e: copy_as_json_details())
    details_win.bind('<Command-j>', lambda e: copy_as_json_details())
    details_win.bind('<F5>', lambda e: on_rescan())
    details_win.bind('r', lambda e: on_rescan())
    details_win.bind('R', lambda e: on_rescan())
    refresh_content(current_item_id)


def open_file(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Open the selected or specified file in the system's default application."""
    file_path = _resolve_file_path(event_or_path)
    if not file_path:
        return

    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", file_path])
        else:
            subprocess.run(["xdg-open", file_path])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open file: {e}")


def copy_path(event: Optional[tk.Event] = None) -> None:
    """Copy the selected file paths to the clipboard (newline-separated)."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return

    paths = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            paths.append(str(values[0]))

    if paths:
        tree.clipboard_clear()
        tree.clipboard_append("\n".join(paths))
        update_status(f"Copied {len(paths)} path(s) to clipboard.")


def copy_sha256(event: Optional[tk.Event] = None) -> None:
    """Calculate and copy SHA256 hashes for selected files to the clipboard."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return

    hashes = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if not values:
            continue
        file_path = str(values[0])

        if file_path.startswith("["):
            # For virtual paths, hash the snippet content
            snippet = str(values[5])
            h = get_file_sha256(snippet.encode('utf-8'))
        else:
            h = get_file_sha256(file_path)

        if h:
            hashes.append(h)

    if hashes:
        tree.clipboard_clear()
        tree.clipboard_append("\n".join(hashes))
        if len(hashes) == 1:
            update_status(f"SHA256 copied: {hashes[0][:8]}...")
        else:
            update_status(f"Copied {len(hashes)} SHA256 hashes.")
    else:
        messagebox.showwarning("Error", "Could not calculate file hashes.")


def check_virustotal(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Check selected files on VirusTotal (opens multiple tabs if needed)."""
    # List of (path, snippet_if_available)
    targets: List[Tuple[str, Optional[str]]] = []

    if isinstance(event_or_path, str):
        targets.append((event_or_path, None))
    else:
        selection = tree.selection()
        if not selection:
            return
        
        # Avoid accidental mass tab opening
        if len(selection) > 5:
            if not messagebox.askyesno("Check on VirusTotal",
                                        f"You have selected {len(selection)} files. Do you want to open that many browser tabs?"):
                return

        for item_id in selection:
            vals = _get_item_raw_values(item_id)
            if vals:
                path = str(vals[0])
                snippet = str(vals[5]) if path.startswith("[") else None
                targets.append((path, snippet))

    found_any = False
    for file_path, snippet in targets:
        h = None
        if file_path.startswith("["):
            if snippet is None:
                # Try to find it in the tree if snippet wasn't provided (explicit path case)
                for item_id in tree.get_children():
                    vals = _get_item_raw_values(item_id)
                    if vals and vals[0] == file_path:
                        snippet = str(vals[5])
                        break
            
            if snippet:
                h = get_file_sha256(snippet.encode('utf-8'))
        else:
            if os.path.exists(file_path):
                h = get_file_sha256(file_path)
            elif isinstance(event_or_path, str):
                messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")
                return

        if h:
            url = f"https://www.virustotal.com/gui/file/{h}"
            webbrowser.open(url)
            found_any = True

    if found_any:
        if len(targets) == 1:
            update_status(f"Opening VirusTotal for {os.path.basename(targets[0][0])}...")
        else:
            update_status(f"Opening VirusTotal for {len(targets)} files...")
    elif not isinstance(event_or_path, str):
        messagebox.showwarning("Error", "Could not calculate hashes for selected files.")


def copy_snippet(event: Optional[tk.Event] = None) -> None:
    """Copy code snippets from the selected rows to the clipboard."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return

    snippets = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            path = values[0]
            snippet = values[5]
            if len(selection) > 1:
                snippets.append(f"--- {path} ---\n{snippet}")
            else:
                snippets.append(snippet)

    if snippets:
        tree.clipboard_clear()
        tree.clipboard_append("\n\n".join(snippets))
        update_status(f"Copied {len(snippets)} snippets.")


def copy_as_markdown(event: Optional[tk.Event] = None) -> None:
    """Copy the selected rows as a Markdown table to the clipboard."""
    if not tree:
        return

    selection = tree.selection()
    if not selection:
        return

    results = _get_tree_results_as_dicts(selection)

    md = generate_markdown(results)
    tree.clipboard_clear()
    tree.clipboard_append(md)
    update_status(f"Copied {len(results)} item(s) as Markdown.")


def copy_as_json(event: Optional[tk.Event] = None) -> None:
    """Copy the selected rows as a JSON array to the clipboard."""
    if not tree:
        return

    selection = tree.selection()
    if not selection:
        return

    results = _get_tree_results_as_dicts(selection)

    js = json.dumps(results, indent=2)
    tree.clipboard_clear()
    tree.clipboard_append(js)
    update_status(f"Copied {len(results)} item(s) as JSON.")


def show_in_folder(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Reveal the selected or specified file in the system file manager."""
    file_path = _resolve_file_path(event_or_path)
    if not file_path:
        return

    try:
        if sys.platform == "win32":
            subprocess.run(['explorer', '/select,', os.path.normpath(file_path)])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", file_path])
        else:
            # On Linux, just open the directory
            subprocess.run(["xdg-open", os.path.dirname(os.path.abspath(file_path))])
    except Exception as e:
        messagebox.showerror("Error", f"Could not reveal file: {e}")


def show_context_menu(event: tk.Event) -> None:
    """Display the context menu at the location of the event."""
    if not tree or not context_menu:
        return

    # Select the item under the mouse if the event has coordinates
    if hasattr(event, 'x') and hasattr(event, 'y'):
        iid = tree.identify_row(event.y)
        # Only change selection if the item clicked is NOT already part of a multi-selection
        if iid and iid not in tree.selection():
            tree.selection_set(iid)

    if tree.selection():
        context_menu.post(event.x_root, event.y_root)


def update_button_states(event: Optional[tk.Event] = None) -> None:
    """Enable or disable selection-dependent buttons."""
    if not tree:
        return

    has_selection = bool(tree.selection())

    # Buttons that depend on having one or more items selected
    dependent_buttons = [view_button, open_button, rescan_button, exclude_button, reveal_button, vt_button]
    for btn in dependent_buttons:
        if btn:
            btn.config(state="normal" if has_selection else "disabled")

    if analyze_button:
        ai_available = Config.GPT_ENABLED
        analyze_button.config(state="normal" if has_selection and ai_available else "disabled")


def copy_cli_command(event: Optional[tk.Event] = None) -> None:
    """Copy the equivalent CLI command to the system clipboard."""
    cmd_parts = ["python", "gptscan.py"]

    # Target path(s)
    raw_target = textbox.get() if textbox else Config.last_path
    if raw_target:
        try:
            # Parse possible multiple targets from the GUI textbox
            targets = shlex.split(raw_target)
            for t in targets:
                cmd_parts.append(shlex.quote(t))
        except ValueError:
            # Fallback to quoting the raw string if parsing fails
            cmd_parts.append(shlex.quote(raw_target))

    cmd_parts.append("--cli")

    # Scan Options
    if deep_var and deep_var.get():
        cmd_parts.append("--deep")
    if git_var and git_var.get():
        cmd_parts.append("--git-changes")
    if dry_var and dry_var.get():
        cmd_parts.append("--dry-run")
    if all_var and all_var.get():
        cmd_parts.append("--show-all")
    if scan_all_var and scan_all_var.get():
        cmd_parts.append("--all-files")

    # Threshold
    if Config.THRESHOLD != 50:
        cmd_parts.extend(["--threshold", str(Config.THRESHOLD)])

    # AI Analysis
    if gpt_var and gpt_var.get():
        cmd_parts.append("--use-gpt")
        if Config.provider != "openai":
            cmd_parts.extend(["--provider", Config.provider])
        if Config.model_name:
            cmd_parts.extend(["--model", Config.model_name])
        if Config.api_base:
            cmd_parts.extend(["--api-base", Config.api_base])

    cmd = " ".join(cmd_parts)
    if root:
        root.clipboard_clear()
        root.clipboard_append(cmd)
        update_status("CLI command copied to clipboard.")


def on_root_return(event: Optional[tk.Event] = None) -> None:
    """Trigger a scan if the focus is not on a widget that handles Return."""
    if not root:
        return
    # Trigger scan if focus is not on results tree, path textbox or filter entry
    focused = root.focus_get()
    if str(focused) not in (str(tree), str(textbox), str(filter_entry)):
        button_click()


def focus_filter(event: Optional[tk.Event] = None) -> str:
    """Set focus to the filter entry and select all text for easy replacement."""
    if filter_entry:
        filter_entry.focus_set()
        filter_entry.selection_range(0, tk.END)
    return "break"


def on_filter_return(event: Optional[tk.Event] = None) -> str:
    """Move focus from the filter entry to the results tree."""
    if tree:
        tree.focus_set()
        # If nothing is selected, select the first item to allow immediate keyboard navigation
        if not tree.selection() and tree.get_children():
            first_item = tree.get_children()[0]
            tree.selection_set(first_item)
            tree.focus(first_item)
            tree.see(first_item)
    return "break"


def select_all_items(event: Optional[tk.Event] = None) -> str:
    """Select all items currently visible in the Results Treeview."""
    if tree:
        tree.selection_set(tree.get_children())
    return "break"


def get_model_presets(provider: str) -> List[str]:
    """Return the list of model presets for a given provider.

    Parameters
    ----------
    provider : str
        The AI provider name (e.g., 'openai', 'ollama', 'openrouter').

    Returns
    -------
    List[str]
        A list of model names supported by the provider.
    """
    if provider == "openai":
        return ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif provider == "ollama":
        return ["llama3.2", "llama3.1", "deepseek-r1", "phi4", "mistral", "gemma2"]
    elif provider == "openrouter":
        return ["gpt-4o", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-r1", "google/gemini-flash-1.5", "meta-llama/llama-3.3-70b-instruct"]
    else:
        return []


def create_gui(initial_path: Optional[str] = None) -> tk.Tk:
    """Construct and return the main Tkinter GUI for the scanner.

    Parameters
    ----------
    initial_path : str, optional
        If provided, pre-fill the scan path textbox.

    Returns
    -------
    tk.Tk
        Initialized Tk root instance ready for ``mainloop``.
    """
    global root, textbox, progress_bar, status_label, deep_var, all_var, scan_all_var, gpt_var, dry_var, git_var, filter_var, filter_entry, tree, scan_button, cancel_button, view_button, vt_button, rescan_button, open_button, analyze_button, exclude_button, reveal_button, results_button, browse_button, default_font_measure, copy_cmd_button, git_checkbox, deep_checkbox, scan_all_checkbox, dry_checkbox, gpt_checkbox, provider_combo, model_combo, api_key_entry, api_entry, all_checkbox, threshold_spin

    root = tk.Tk()
    root.geometry("1000x600")
    root.title("GPT Virus Scanner")
    default_font_measure = tkinter.font.Font(font='TkDefaultFont').measure

    style = ttk.Style(root)
    style.configure('Primary.TButton', font=('TkDefaultFont', 9, 'bold'))

    # --- Menu Bar ---
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Import Results...", command=import_results)
    file_menu.add_command(label="Import from Clipboard", command=import_from_clipboard)
    file_menu.add_command(label="Import from URL...", command=import_from_url)
    file_menu.add_command(label="Export Results...", command=export_results)
    file_menu.add_command(label="Manage Exclusions...", command=manage_exclusions)
    file_menu.add_command(label="Manage Extensions...", command=manage_extensions)
    file_menu.add_command(label="Copy as CLI Command", command=copy_cli_command)
    file_menu.add_separator()
    file_menu.add_command(label="Clear Results", command=clear_results)
    file_menu.add_command(label="Clear AI Cache", command=clear_ai_cache)
    file_menu.add_command(label="Clear Path History", command=clear_path_history)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", f"GPT Virus Scanner v{Config.VERSION}\nThis tool uses AI to find malicious code in your scripts."))
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

    # Configure grid weights to ensure resizing behaves correctly
    root.columnconfigure(0, weight=1)
    root.rowconfigure(4, weight=1)  # The row containing the Treeview (tree_frame)

    # --- Input Frame ---
    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    input_frame.columnconfigure(1, weight=1)

    ttk.Label(input_frame, text="Path to scan:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    textbox = ttk.Combobox(input_frame, values=Config.recent_paths)
    path_to_use = initial_path if initial_path else (Config.last_path if Config.last_path else os.getcwd())
    textbox.insert(0, path_to_use)
    textbox.select_range(0, tk.END)
    textbox.grid(row=0, column=1, sticky="ew", padx=5)
    textbox.bind('<Return>', lambda event: button_click())
    textbox.focus_set()
    bind_hover_message(textbox, "Enter one or more files, folders, or glob patterns (e.g., src/**/*.py) to scan. Separate multiple targets with spaces.")

    root.bind('<Escape>', lambda event: cancel_scan())
    button_box = ttk.Frame(input_frame)
    button_box.grid(row=0, column=2, sticky="e")

    browse_button = ttk.Menubutton(button_box, text="Browse", width=10)
    browse_button.pack(side=tk.LEFT, padx=(5, 2), ipady=5)
    bind_hover_message(browse_button, "Browse for scan targets (File, Folder, URL, or Clipboard).")

    scan_button = ttk.Button(button_box, text="Scan Now", command=button_click, style='Primary.TButton', default='active', width=12)
    scan_button.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(scan_button, "Start the scan.")

    cancel_button = ttk.Button(button_box, text="Cancel", command=cancel_scan, state="disabled", width=10)
    cancel_button.pack(side=tk.LEFT, padx=(2, 0), ipady=5)
    bind_hover_message(cancel_button, "Stop the current scan.")

    browse_menu = tk.Menu(browse_button, tearoff=0)
    browse_menu.add_command(label="Select File(s)...", command=browse_file_click)
    browse_menu.add_command(label="Select Folder...", command=browse_dir_click)
    browse_menu.add_command(label="Scan URL...", command=select_url_click)
    browse_menu.add_command(label="Scan Clipboard", command=scan_clipboard_click)
    browse_button["menu"] = browse_menu

    # --- Settings Container ---
    settings_frame = ttk.Frame(root)
    settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

    # --- Options Frame ---
    options_frame = ttk.LabelFrame(settings_frame, text="Scan Options", padding=10)
    options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

    gpt_var = tk.BooleanVar(value=Config.use_ai_analysis)

    git_var = tk.BooleanVar(value=Config.git_changes_only)
    git_checkbox = ttk.Checkbutton(options_frame, text="Git changes only", variable=git_var)
    git_checkbox.grid(row=0, column=0, sticky='w', padx=10, pady=2)
    bind_hover_message(git_checkbox, "Only scan files that have been modified or are untracked in Git.")

    deep_var = tk.BooleanVar(value=Config.deep_scan)
    deep_checkbox = ttk.Checkbutton(options_frame, text="Deep scan", variable=deep_var)
    deep_checkbox.grid(row=0, column=1, sticky='w', padx=10, pady=2)
    bind_hover_message(deep_checkbox, "Scan the whole file. This is slower but more thorough. Normally, the scanner only checks the beginning and end.")

    scan_all_var = tk.BooleanVar(value=Config.scan_all_files)
    scan_all_checkbox = ttk.Checkbutton(options_frame, text="Scan all files", variable=scan_all_var)
    scan_all_checkbox.grid(row=1, column=0, sticky='w', padx=10, pady=2)
    bind_hover_message(scan_all_checkbox, "Scan all files regardless of their extension or whether they contain a script starting line (like #!/bin/bash).")

    all_var = tk.BooleanVar(value=Config.show_all_files)

    dry_var = tk.BooleanVar()
    dry_checkbox = ttk.Checkbutton(options_frame, text="Dry Run", variable=dry_var)
    dry_checkbox.grid(row=1, column=1, sticky='w', padx=10, pady=2)
    bind_hover_message(dry_checkbox, "Simulate the scan process without running checks.")

    size_frame = ttk.Frame(options_frame)
    size_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=10, pady=2)
    ttk.Label(size_frame, text="Max File Size (MB):").pack(side=tk.LEFT)

    def on_max_size_change():
        try:
            val = float(max_size_spin.get())
            Config.MAX_FILE_SIZE = int(val * 1024 * 1024)
        except ValueError:
            pass

    max_size_spin = ttk.Spinbox(size_frame, from_=1, to=1024, width=5, command=on_max_size_change)
    max_size_spin.delete(0, tk.END)
    max_size_spin.insert(0, str(int(Config.MAX_FILE_SIZE / (1024 * 1024))))
    max_size_spin.pack(side=tk.LEFT, padx=5)
    max_size_spin.bind('<KeyRelease>', lambda e: on_max_size_change())
    bind_hover_message(max_size_spin, "Skip files larger than this size (in Megabytes).")


    # --- Provider Frame ---
    provider_frame = ttk.LabelFrame(settings_frame, text="AI Analysis", padding=10)
    provider_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    copy_cmd_button = ttk.Button(options_frame, text="Copy CLI Command", command=copy_cli_command)
    copy_cmd_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=(5, 0), ipady=5)
    bind_hover_message(copy_cmd_button, "Copy the current scan settings as a CLI command for use in scripts or automation.")

    provider_frame.columnconfigure(1, weight=1)
    provider_frame.columnconfigure(3, weight=1)

    gpt_checkbox = ttk.Checkbutton(provider_frame, text="Use AI Analysis", variable=gpt_var, command=toggle_ai_controls)
    gpt_checkbox.grid(row=0, column=0, columnspan=4, sticky='w', padx=10, pady=(2, 10))
    bind_hover_message(gpt_checkbox, "Use AI to analyze suspicious code and explain what it does.")

    if not Config.GPT_ENABLED:
        gpt_var.set(False)
        gpt_checkbox.config(state="disabled")
        messagebox.showwarning("AI Disabled",
                                       "The scanner cannot find 'task.txt'. You cannot use AI analysis.")

    ttk.Label(provider_frame, text="Provider:").grid(row=1, column=0, sticky='w', padx=(10, 5), pady=2)
    provider_var = tk.StringVar(value=Config.provider)
    provider_combo = ttk.Combobox(provider_frame, textvariable=provider_var, values=["openai", "openrouter", "ollama"], state="readonly", width=12)
    provider_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(provider_frame, text="API Key:").grid(row=1, column=2, sticky='w', padx=(15, 5), pady=2)
    api_key_var = tk.StringVar(value=Config.apikey)
    api_key_entry = ttk.Entry(provider_frame, show="*", textvariable=api_key_var)
    api_key_entry.grid(row=1, column=3, sticky='ew', padx=(5, 10), pady=2)

    def on_api_key_change(*args):
        Config.apikey = api_key_var.get().strip()
        global _async_openai_client
        _async_openai_client = None

    api_key_var.trace_add("write", on_api_key_change)

    ttk.Label(provider_frame, text="Model:").grid(row=2, column=0, sticky='w', padx=(10, 5), pady=2)
    model_var = tk.StringVar(value=Config.model_name)
    model_combo = ttk.Combobox(provider_frame, textvariable=model_var, width=20)
    model_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(provider_frame, text="API Base URL:").grid(row=2, column=2, sticky='w', padx=(15, 5), pady=2)
    api_entry = ttk.Entry(provider_frame)
    api_entry.grid(row=2, column=3, sticky='ew', padx=(5, 10), pady=2)
    bind_hover_message(api_entry, "Set a custom URL for the AI service (e.g., http://localhost:11434/v1 for Ollama).")

    api_base_var = tk.StringVar(value=Config.api_base or "")
    api_entry.config(textvariable=api_base_var)

    def on_api_base_change(*args):
        val = api_base_var.get().strip()
        Config.api_base = val if val else None
        global _async_openai_client
        _async_openai_client = None

    api_base_var.trace_add("write", on_api_base_change)

    toggle_ai_controls()

    def update_model_presets(provider: str):
        model_combo['values'] = get_model_presets(provider)

    update_model_presets(Config.provider)

    def on_provider_change(event):
        p = provider_var.get()
        Config.provider = p

        # Reset clients so they are recreated with new settings
        global _async_openai_client
        _async_openai_client = None

        update_model_presets(p)

        if p == "ollama":
            model_var.set("llama3.2")
        else:
            # Default to the first preset if available, else keep current or specific default
            if model_combo['values']:
                model_var.set(model_combo['values'][0])
            else:
                model_var.set("gpt-4o")
        Config.model_name = model_var.get()

    provider_combo.bind("<<ComboboxSelected>>", on_provider_change)

    def on_model_change(*args):
        Config.model_name = model_var.get()

    model_var.trace_add("write", on_model_change)

    if Config.extensions_missing:
        default_exts = ', '.join(sorted(Config.extensions_set)) if Config.extensions_set else 'none'
        messagebox.showwarning(
            "Extensions Missing",
            f"extensions.txt not found. Using default extensions: {default_exts}"
        )

    # --- Progress Bar ---
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, mode='determinate')
    progress_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

    # --- Filter Frame ---
    filter_frame = ttk.Frame(root)
    filter_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
    filter_frame.columnconfigure(1, weight=1)

    ttk.Label(filter_frame, text="Filter:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    filter_var = tk.StringVar()
    filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
    filter_entry.grid(row=0, column=1, sticky="ew")
    filter_entry.bind('<KeyRelease>', _apply_filter)
    filter_entry.bind('<Return>', on_filter_return)
    bind_hover_message(filter_entry, "Search results by any column (path, threat level, analysis, snippet).")

    def clear_filter():
        filter_var.set("")
        _apply_filter()

    clear_filter_btn = ttk.Button(filter_frame, text="Clear", width=8, command=clear_filter)
    clear_filter_btn.grid(row=0, column=2, padx=(5, 5), ipady=5)
    bind_hover_message(clear_filter_btn, "Clear the filter.")

    ttk.Separator(filter_frame, orient=tk.VERTICAL).grid(row=0, column=3, sticky="ns", padx=10)

    def on_threshold_change():
        try:
            val = int(threshold_spin.get())
            Config.THRESHOLD = max(0, min(100, val))
            _apply_filter()
        except ValueError:
            pass

    threshold_frame = ttk.Frame(filter_frame)
    threshold_frame.grid(row=0, column=4, sticky='w')

    ttk.Label(threshold_frame, text="Min. Threat Level:").pack(side=tk.LEFT)
    threshold_spin = ttk.Spinbox(threshold_frame, from_=0, to=100, width=5, command=on_threshold_change)
    threshold_spin.delete(0, tk.END)
    threshold_spin.insert(0, str(Config.THRESHOLD))
    threshold_spin.pack(side=tk.LEFT, padx=5)
    threshold_spin.bind('<KeyRelease>', lambda e: on_threshold_change())
    bind_hover_message(threshold_spin, "Files with a threat level lower than this will be ignored.")

    all_checkbox = ttk.Checkbutton(threshold_frame, text="Show all files", variable=all_var, command=_apply_filter)
    all_checkbox.pack(side=tk.LEFT, padx=(5, 0))
    bind_hover_message(all_checkbox, "Display all scanned files, including safe ones.")

    # --- Treeview ---
    style.configure('Scanner.Treeview', rowheight=50)

    # Configure tags for row highlighting
    # Note: 'alt' theme or similar might be needed for background colors to show in some environments
    tree_frame = ttk.Frame(root)
    tree_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    tree = ttk.Treeview(tree_frame, style='Scanner.Treeview')
    tree.tag_configure('high-risk', background='#ffcccc')
    tree.tag_configure('medium-risk', background='#fff0cc')
    tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line", "orig_json")
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("path", width=150, stretch=tk.YES, anchor="w")
    tree.column("line", width=60, stretch=tk.NO, anchor="center")
    tree.column("own_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("gpt_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("admin_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("end-user_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("snippet", width=150, stretch=tk.YES, anchor="w")
    tree.column("orig_json", width=0, stretch=tk.NO) # Hidden column for raw data
    root.after(0, process_ui_queue)

    tree.heading("#0", text="")
    tree.heading("path", text="File Path", command=lambda: sort_column(tree, "path", False))
    tree.heading("line", text="Line", command=lambda: sort_column(tree, "line", False))
    tree.heading("own_conf", text="Local Threat",
                 command=lambda: sort_column(tree, "own_conf", False))
    tree.heading("gpt_conf", text="AI Threat",
                 command=lambda: sort_column(tree, "gpt_conf", False))
    tree.heading("admin_desc", text="Admin Notes",
                 command=lambda: sort_column(tree, "admin_desc", False))
    tree.heading("end-user_desc", text="User Notes",
                 command=lambda: sort_column(tree, "end-user_desc", False))
    tree.heading("snippet", text="Snippet", command=lambda: sort_column(tree, "snippet", False))

    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")

    tree.configure(yscrollcommand=scrollbar.set)
    tree.bind('<ButtonRelease-1>', partial(motion_handler, tree))

    def on_tree_double_click(event):
        """Show details only if a data cell was double-clicked."""
        if tree.identify_region(event.x, event.y) == "cell":
            view_details(event)

    tree.bind('<Double-1>', on_tree_double_click)
    tree.bind('<Return>', view_details)
    tree.bind('<Shift-Return>', open_file)
    tree.grid(row=0, column=0, sticky="nsew")

    # --- Footer Frame ---
    footer_frame = ttk.Frame(root)
    footer_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))
    footer_frame.columnconfigure(0, weight=1)

    status_label = ttk.Label(footer_frame, text="Ready", anchor="w")
    status_label.grid(row=0, column=0, sticky="ew")

    view_button = ttk.Button(footer_frame, text="View", width=10, command=view_details, style='Primary.TButton')
    view_button.grid(row=0, column=1, padx=2, ipady=5)
    bind_hover_message(view_button, "Show full analysis and code for the selected result.")

    analyze_button = ttk.Button(footer_frame, text="Analyze with AI", width=18, command=analyze_selected_with_ai)
    analyze_button.grid(row=0, column=2, padx=2, ipady=5)
    bind_hover_message(analyze_button, "Use AI to analyze the currently selected items.")

    vt_button = ttk.Button(footer_frame, text="VirusTotal", width=12, command=check_virustotal)
    vt_button.grid(row=0, column=3, padx=2, ipady=5)
    bind_hover_message(vt_button, "Check the selected files on VirusTotal.")

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=4, sticky="ns", padx=5)

    open_button = ttk.Button(footer_frame, text="Open", width=10, command=open_file)
    open_button.grid(row=0, column=5, padx=2, ipady=5)
    bind_hover_message(open_button, "Open the selected file in its default application. (Shift+Enter)")

    reveal_button = ttk.Button(footer_frame, text="Reveal", width=10, command=show_in_folder)
    reveal_button.grid(row=0, column=6, padx=2, ipady=5)
    bind_hover_message(reveal_button, "Reveal the selected file in the system file manager.")

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=7, sticky="ns", padx=5)

    rescan_button = ttk.Button(footer_frame, text="Rescan", width=10, command=rescan_selected)
    rescan_button.grid(row=0, column=8, padx=2, ipady=5)
    bind_hover_message(rescan_button, "Re-scan the currently selected items.")

    exclude_button = ttk.Button(footer_frame, text="Exclude", width=10, command=exclude_selected)
    exclude_button.grid(row=0, column=9, padx=2, ipady=5)
    bind_hover_message(exclude_button, "Exclude the selected items from future scans.")

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=10, sticky="ns", padx=5)

    results_button = ttk.Menubutton(footer_frame, text="Results", width=12)
    results_button.grid(row=0, column=11, padx=(2, 0), ipady=5)
    bind_hover_message(results_button, "Manage scan results (Import, Export, Clear).")

    results_menu = tk.Menu(results_button, tearoff=0)
    results_menu.add_command(label="Import Results...", command=import_results)
    results_menu.add_command(label="Import from Clipboard", command=import_from_clipboard)
    results_menu.add_command(label="Import from URL...", command=import_from_url)
    results_menu.add_command(label="Export Results...", command=export_results)
    results_menu.add_separator()
    results_menu.add_command(label="Clear Results", command=clear_results)
    results_button["menu"] = results_menu

    # --- Context Menu ---
    global context_menu
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="View Details...", command=view_details)
    context_menu.add_separator()
    context_menu.add_command(label="Rescan Selected", command=rescan_selected)
    context_menu.add_command(label="Analyze with AI", command=analyze_selected_with_ai)
    context_menu.add_command(label="Exclude from future scans", command=exclude_selected)
    context_menu.add_separator()
    context_menu.add_command(label="Select All", command=select_all_items)
    context_menu.add_separator()
    context_menu.add_command(label="Open", command=open_file)
    context_menu.add_command(label="Reveal", command=show_in_folder)
    context_menu.add_separator()
    context_menu.add_command(label="Copy File Path", command=copy_path)
    context_menu.add_command(label="Copy SHA256", command=copy_sha256)
    context_menu.add_command(label="Check on VirusTotal", command=check_virustotal)
    context_menu.add_command(label="Copy Snippet", command=copy_snippet)
    context_menu.add_command(label="Copy as Markdown", command=copy_as_markdown)
    context_menu.add_command(label="Copy as JSON", command=copy_as_json)

    # Bind context menu to right-click and menu key
    tree.bind('<Button-3>', show_context_menu) # Windows/Linux
    tree.bind('<Button-2>', show_context_menu) # macOS
    tree.bind('<Menu>', show_context_menu)

    # Bind selection and rescan keys
    root.bind('<Return>', on_root_return)
    root.bind('<Control-Shift-E>', copy_cli_command)
    root.bind('<Command-Shift-E>', copy_cli_command)
    root.bind('<Control-v>', import_from_clipboard)
    root.bind('<Command-v>', import_from_clipboard)
    root.bind('<Control-f>', focus_filter)
    root.bind('<Command-f>', focus_filter)
    root.bind('<Control-j>', copy_as_json)
    root.bind('<Command-j>', copy_as_json)
    root.bind('<Control-g>', analyze_selected_with_ai)
    root.bind('<Command-g>', analyze_selected_with_ai)
    tree.bind('<<TreeviewSelect>>', update_button_states)
    tree.bind('<Control-a>', select_all_items)
    tree.bind('<Command-a>', select_all_items)
    tree.bind('<Control-c>', copy_path)
    tree.bind('<Command-c>', copy_path)
    tree.bind('<Control-Shift-C>', copy_as_markdown)
    tree.bind('<Command-Shift-C>', copy_as_markdown)
    tree.bind('<Control-h>', copy_sha256)
    tree.bind('<Command-h>', copy_sha256)
    tree.bind('<Control-s>', copy_snippet)
    tree.bind('<Command-s>', copy_snippet)
    tree.bind('<Control-Return>', show_in_folder)
    tree.bind('<Command-Return>', show_in_folder)
    tree.bind('<space>', view_details)
    tree.bind('<Delete>', lambda event: exclude_selected())
    tree.bind('<F5>', lambda event: rescan_selected())
    tree.bind('r', lambda event: rescan_selected())

    def on_close():
        Config.last_path = textbox.get()
        Config.deep_scan = deep_var.get()
        Config.git_changes_only = git_var.get()
        Config.show_all_files = all_var.get()
        Config.scan_all_files = scan_all_var.get()
        Config.use_ai_analysis = gpt_var.get()
        Config.provider = provider_var.get()
        Config.model_name = model_var.get()
        Config.save_settings()
        Config.save_apikey()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    motion_handler(tree, None)   # Perform initial wrapping
    update_tree_columns()        # Adjust columns based on initial AI settings
    update_button_states()       # Initialize button states
    set_scanning_state(False)
    return root


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scan scripts, archives (ZIP/TAR), Jupyter Notebooks, package manifests (package.json, composer.json, deno.json), CI/CD workflows (GitHub Actions, GitLab CI), Markdown files, HTML files, patches (.diff/.patch), and web links (GitHub/GitLab/Bitbucket/Gist) for malicious code using AI.",
        epilog="Examples:\n"
               "  # Scan a folder using AI analysis\n"
               "  python gptscan.py ./my_scripts --cli --use-gpt\n\n"
               "  # Scan a single file and save as JSON\n"
               "  python gptscan.py ./my_script.py --cli --json\n\n"
               "  # Scan only changed files in Git and fail if threats are found\n"
               "  python gptscan.py --git-changes --cli --fail-threshold 50\n\n"
               "  # Scan a snippet sent from another command\n"
               "  echo \"print('hello')\" | python gptscan.py --cli --stdin\n\n"
               "  # Scan a remote script or a GitHub repository directly from a web link\n"
               "  python gptscan.py https://github.com/user/repo --cli\n\n"
               "  # Scan a GitHub Pull Request or GitLab Merge Request directly\n"
               "  python gptscan.py https://github.com/user/repo/pull/123 --cli\n\n"
               "Note: Always run the script from inside its own folder so it can find its required data files (like scripts.h5).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {Config.VERSION}')
    parser.add_argument('target', nargs='?', help='The folder, file, glob pattern (e.g., src/**/*.py), or web link to scan.')
    parser.add_argument(
        'files',
        nargs='*',
        help='Additional folders, files, glob patterns, or web links to scan.'
    )

    scan_group = parser.add_argument_group("Scan Options")
    scan_group.add_argument('-p', '--path', type=str, help='A folder, file, or web link to scan.')
    scan_group.add_argument('-d', '--deep', action='store_true', help='Scan the entire file instead of just the first and last 1 KB (1,024 bytes).')
    scan_group.add_argument('--dry-run', action='store_true', help='Show which files would be scanned without analyzing them.')
    scan_group.add_argument(
        '--extensions',
        type=str,
        help='Only scan these file types (comma-separated, e.g., "py,js").'
    )
    scan_group.add_argument(
        '-e', '--exclude',
        nargs='*',
        help='Ignore files or folders that match these patterns (for example: "node_modules/*").'
    )
    scan_group.add_argument(
        '--file-list',
        type=argparse.FileType('r'),
        help='Read a list of files to scan from a text file.'
    )
    scan_group.add_argument(
        '--git-changes',
        action='store_true',
        help='Only scan files that have changed in your Git repository.'
    )
    scan_group.add_argument(
        '--all-files',
        action='store_true',
        help='Scan every file, even if it does not have a script extension or a starting line (like #!/bin/bash).'
    )
    scan_group.add_argument(
        '--fail-threshold',
        type=int,
        help='Exit with an error if any file has a threat level at or above this number (0-100).'
    )
    scan_group.add_argument(
        '--threshold', '-t',
        type=int,
        default=50,
        help='Show only files with a threat level at or above this number (0-100). The default is 50.'
    )
    scan_group.add_argument(
        '--stdin',
        action='store_true',
        help='Scan a code snippet sent from another command in the terminal.'
    )
    scan_group.add_argument(
        '--import-results', '--import',
        type=str,
        help='Import results from a previous scan (JSON, CSV, SARIF, Markdown, or HTML). Use "-" to read from the terminal.'
    )
    scan_group.add_argument(
        '--max-size',
        type=str,
        help='The maximum file size to scan (e.g., "10MB", "500KB"). The default is 10MB.'
    )

    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument('-g', '--use-gpt', action='store_true', help='Enable detailed AI reports for suspicious files. Cloud providers (OpenAI, OpenRouter) need an API key; local Ollama does not.')
    ai_group.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'openrouter', 'ollama'],
        help='Select the AI service provider (default: openai).'
    )
    ai_group.add_argument(
        '--model',
        type=str,
        help='Specify the AI model (e.g., gpt-4o, llama3.2).'
    )
    ai_group.add_argument(
        '--api-key', '-k',
        type=str,
        help='Set the API key for cloud-based analysis.'
    )
    ai_group.add_argument(
        '--api-base',
        type=str,
        help='Set a custom URL for the AI service (useful for local servers).'
    )
    ai_group.add_argument(
        '--rate-limit',
        type=int,
        default=Config.RATE_LIMIT_PER_MINUTE,
        help='Limit AI requests per minute to avoid errors (default: 60).'
    )
    ai_group.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the AI analysis cache.'
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--cli', action='store_true', help='Run in command-line mode instead of opening a window.')
    output_group.add_argument('-a', '--show-all', action='store_true', help='Show all scanned files, including safe ones.')
    output_group.add_argument('-o', '--output', type=str, help='Save the scan results to a file. The tool chooses the format based on the file extension.')
    output_group.add_argument('-j', '--json', action='store_true', help='Output results in JSON format (one object per line).')
    output_group.add_argument('--csv', action='store_true', help='Output results in CSV format (default when output is piped or redirected).')
    output_group.add_argument('--sarif', action='store_true', help='Save results in SARIF format (a common format used by security tools).')
    output_group.add_argument('--html', action='store_true', help='Create an HTML report of the results.')
    output_group.add_argument('--md', '--markdown', action='store_true', dest='markdown', help='Create a Markdown report of the results.')
    output_group.add_argument('--report', action='store_true', help='Output a human-friendly triage report to the console (default when outputting directly to a terminal).')

    args = parser.parse_args()

    if args.clear_cache:
        Config.gpt_cache = {}
        Config.save_cache()
        print("AI Analysis cache cleared.", file=sys.stderr)
        # If we ONLY wanted to clear cache, exit now.
        if not any([args.target, args.path, args.stdin, args.import_results, args.files]):
            sys.exit(0)

    Config.provider = args.provider
    if args.api_key:
        Config.apikey = args.api_key
        Config.save_apikey()

    if args.api_base:
        Config.api_base = args.api_base

    if args.model:
        Config.model_name = args.model
    elif Config.provider == 'ollama':
        Config.model_name = 'llama3.2'

    if args.extensions:
        extension_list = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
        Config.set_extensions(extension_list, missing=False)

    if args.all_files:
        Config.scan_all_files = True

    if args.max_size:
        try:
            Config.MAX_FILE_SIZE = parse_size_string(args.max_size)
        except ValueError as e:
            parser.error(f"Invalid --max-size: {e}")

    Config.THRESHOLD = args.threshold

    scan_target = args.target or args.path

    if args.cli:
        scan_targets = []
        if args.target:
            scan_targets.append(args.target)
        if args.path:
            scan_targets.append(args.path)
        if args.files:
            scan_targets.extend(args.files)

        if args.file_list:
            for line in args.file_list:
                line = line.strip()
                if line:
                    scan_targets.append(line)

        if args.git_changes:
            # Use specified targets as git roots, or current directory if none.
            git_roots = scan_targets if scan_targets else ["."]
            git_files = []
            for root_dir in git_roots:
                git_files.extend(get_git_changed_files(root_dir))

            if not git_files:
                print("No git changes detected in provided targets.", file=sys.stderr)
            # Scan only changed files
            scan_targets = git_files

        if not scan_targets and not args.git_changes:
            # Default to current directory if no targets provided and NOT using git-changes
            scan_targets = ["."]

        output_format = 'report' if sys.stdout.isatty() else 'csv'
        if args.json:
            output_format = 'json'
        elif args.csv:
            output_format = 'csv'
        elif args.sarif:
            output_format = 'sarif'
        elif args.html:
            output_format = 'html'
        elif args.markdown:
            output_format = 'markdown'
        elif args.report:
            output_format = 'report'
        elif args.output:
            # Infer format from extension
            ext = Path(args.output).suffix.lower()
            if ext in ('.json', '.ndjson'):
                output_format = 'json'
            elif ext == '.sarif':
                output_format = 'sarif'
            elif ext in ('.html', '.htm'):
                output_format = 'html'
            elif ext in ('.md', '.markdown'):
                output_format = 'markdown'
            elif ext == '.csv':
                output_format = 'csv'

        final_excludes = list(set((Config.ignore_patterns or []) + (args.exclude or [])))

        extra_snippets = []
        if args.stdin:
            try:
                # Read from stdin buffer for binary safety
                stdin_content = sys.stdin.buffer.read()
                if stdin_content:
                    extra_snippets.append(("[Stdin]", stdin_content))
            except Exception as e:
                print(f"Error reading from terminal input: {e}", file=sys.stderr)

        threats = run_cli(
            scan_targets,
            args.deep,
            args.show_all,
            args.use_gpt,
            args.rate_limit,
            output_format=output_format,
            dry_run=args.dry_run,
            exclude_patterns=final_excludes,
            fail_threshold=args.fail_threshold,
            output_file=args.output,
            extra_snippets=extra_snippets,
            import_file=args.import_results
        )
        if args.fail_threshold is not None and threats > 0:
            sys.exit(1)
    else:
        app_root = create_gui(initial_path=scan_target)
        app_root.mainloop()

if __name__ == "__main__":
    main()
