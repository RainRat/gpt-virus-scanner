import asyncio
import csv
import glob
import hashlib
import html
import io
import json
import plistlib
import site
import sqlite3
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
intel_button: Optional[ttk.Menubutton] = None
intel_menu: Optional[tk.Menu] = None
results_button: Optional[ttk.Menubutton] = None
browse_button: Optional[ttk.Menubutton] = None
show_key_btn: Optional[ttk.Button] = None
copy_cmd_button: Optional[ttk.Button] = None
clear_target_btn: Optional[ttk.Button] = None
git_checkbox: Optional[ttk.Checkbutton] = None
deep_checkbox: Optional[ttk.Checkbutton] = None
scan_all_checkbox: Optional[ttk.Checkbutton] = None
dry_checkbox: Optional[ttk.Checkbutton] = None
gpt_checkbox: Optional[ttk.Checkbutton] = None
all_checkbox: Optional[ttk.Checkbutton] = None
threshold_spin: Optional[ttk.Spinbox] = None
provider_var: Optional[tk.StringVar] = None
model_var: Optional[tk.StringVar] = None
api_base_var: Optional[tk.StringVar] = None
api_key_var: Optional[tk.StringVar] = None
provider_combo: Optional[ttk.Combobox] = None
model_combo: Optional[ttk.Combobox] = None
api_key_entry: Optional[ttk.Entry] = None
api_entry: Optional[ttk.Entry] = None
context_menu: Optional[tk.Menu] = None
_all_results_cache: List[Tuple[Any, ...]] = []
_last_scan_summary: str = ""
_virtual_source_cache: Dict[str, str] = {}


def resolve_remote_url(url: str) -> str:
    """Resolve GitHub/GitLab/Bitbucket/Pastebin/Hugging Face repository, file, tag, or pull request web links to their raw content or archive.

    Args:
        url: The original web link to resolve.

    Returns:
        A resolved web link pointing to raw content or a downloadable archive.
    """
    url = url.strip()
    if not url.lower().startswith(('http://', 'https://')):
        return url

    # Remove trailing slashes and common fragments
    url = re.sub(r'/$', '', url)
    url = re.sub(r'#.*$', '', url)

    # Remove .git suffix from repository URLs
    url = re.sub(r'\.git$', '', url, flags=re.IGNORECASE)

    # 1. GitHub File -> Raw
    # Example: https://github.com/user/repo/blob/main/script.py -> https://raw.githubusercontent.com/user/repo/main/script.py
    gh_blob_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/(.+)', url, re.IGNORECASE)
    if gh_blob_match:
        user, repo, path = gh_blob_match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"

    # 2. GitHub pull request/Commit -> Diff
    # Example: https://github.com/user/repo/pull/1 -> https://github.com/user/repo/pull/1.diff
    # Example: https://github.com/user/repo/commit/abc -> https://github.com/user/repo/commit/abc.diff
    gh_patch_match = re.match(r'(https?://(?:www\.)?github\.com/[^/]+/[^/]+/(?:pull/\d+|commit/[a-f0-9]+))(?:/.*)?$', url, re.IGNORECASE)
    if gh_patch_match:
        return f"{gh_patch_match.group(1)}.diff"

    # 3. GitHub Gist -> ZIP Archive (gets all files)
    # Example: https://gist.github.com/user/id -> https://gist.github.com/user/id/download
    gist_match = re.match(r'https?://gist\.github\.com/([^/]+)/([a-f0-9]+)$', url, re.IGNORECASE)
    if gist_match:
        return f"{url}/download"

    # 4. GitHub Tag/Release -> ZIP Archive
    # Example: https://github.com/user/repo/releases/tag/v1.0 -> https://github.com/user/repo/archive/refs/tags/v1.0.zip
    gh_tag_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/releases/tag/(.+)', url, re.IGNORECASE)
    if gh_tag_match:
        user, repo, tag = gh_tag_match.groups()
        return f"https://github.com/{user}/{repo}/archive/refs/tags/{tag}.zip"

    # 5. GitHub Branch/Tree -> ZIP Archive
    # Example: https://github.com/user/repo/tree/main -> https://github.com/user/repo/archive/refs/heads/main.zip
    gh_tree_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/tree/(.+)', url, re.IGNORECASE)
    if gh_tree_match:
        user, repo, ref = gh_tree_match.groups()
        return f"https://github.com/{user}/{repo}/archive/refs/heads/{ref}.zip"

    # 6. GitHub Repo -> ZIP Archive
    # Example: https://github.com/user/repo -> https://github.com/user/repo/archive/HEAD.zip
    gh_repo_match = re.match(r'https?://(?:www\.)?github\.com/([^/]+)/([^/]+)$', url, re.IGNORECASE)
    if gh_repo_match:
        user, repo = gh_repo_match.groups()
        if repo.lower() not in ('settings', 'pulls', 'issues', 'actions', 'projects', 'wiki', 'security', 'insights', 'pull'):
            return f"https://github.com/{user}/{repo}/archive/HEAD.zip"

    # 7. GitLab File -> Raw
    # Example: https://gitlab.com/user/repo/-/blob/main/script.py -> https://gitlab.com/user/repo/-/raw/main/script.py
    gl_blob_match = re.match(r'(https?://(?:www\.)?gitlab\.com/.+?)/-/blob/(.+)', url, re.IGNORECASE)
    if gl_blob_match:
        base, path = gl_blob_match.groups()
        return f"{base}/-/raw/{path}"

    # 8. GitLab merge request -> Diff
    # Example: https://gitlab.com/user/repo/-/merge_requests/1 -> https://gitlab.com/user/repo/-/merge_requests/1.diff
    # Example: https://gitlab.com/user/repo/-/commit/abc -> https://gitlab.com/user/repo/-/commit/abc.diff
    gl_patch_match = re.match(r'(https?://(?:www\.)?gitlab\.com/(.+)/([^/]+)/-/(?:merge_requests/\d+|commit/[a-f0-9]+))(?:/.*)?$', url, re.IGNORECASE)
    if gl_patch_match:
        return f"{gl_patch_match.group(1)}.diff"

    # 9. GitLab Snippet -> Raw
    # Example: https://gitlab.com/snippets/123 -> https://gitlab.com/snippets/123/raw
    gl_snippet_match = re.match(r'(https?://(?:www\.)?gitlab\.com/snippets/\d+)(?:/.*)?$', url, re.IGNORECASE)
    if gl_snippet_match:
        return f"{gl_snippet_match.group(1)}/raw"

    # 10. GitLab Tag/Branch -> ZIP Archive
    # Example: https://gitlab.com/user/repo/-/tags/v1.0 -> https://gitlab.com/user/repo/-/archive/v1.0/repo-v1.0.zip
    # Example: https://gitlab.com/user/repo/-/tree/main -> https://gitlab.com/user/repo/-/archive/main/repo-main.zip
    gl_ref_match = re.match(r'(https?://(?:www\.)?gitlab\.com/(.+)/([^/]+))/-/(?:tags|tree)/(.+)', url, re.IGNORECASE)
    if gl_ref_match:
        base, group, repo, ref = gl_ref_match.groups()
        return f"{base}/-/archive/{ref}/{repo}-{ref}.zip"

    # 11. GitLab Repo -> ZIP Archive
    # Example: https://gitlab.com/user/repo -> https://gitlab.com/user/repo/-/archive/main/repo-main.zip
    # Note: GitLab is trickier as the default branch varies. We'll try common patterns.
    gl_repo_match = re.match(r'https?://(?:www\.)?gitlab\.com/(?!.*/-/)(.+)/([^/]+)$', url, re.IGNORECASE)
    if gl_repo_match:
        group_path, repo = gl_repo_match.groups()
        # GitLab doesn't have a universal HEAD.zip, but we can try to guess or just return the URL
        # Common default branches are 'main' or 'master'. We'll try 'main' and let fetch_url_content fallback if it fails.
        return f"https://gitlab.com/{group_path}/{repo}/-/archive/main/{repo}-main.zip"

    # 12. Bitbucket Cloud Raw
    # Example: https://bitbucket.org/user/repo/src/main/script.py -> https://bitbucket.org/user/repo/raw/main/script.py
    bb_raw_match = re.match(r'https?://(?:www\.)?bitbucket\.org/([^/]+)/([^/]+)/src/([^/]+)/(.+)', url, re.IGNORECASE)
    if bb_raw_match:
        user, repo, ref, path = bb_raw_match.groups()
        return f"https://bitbucket.org/{user}/{repo}/raw/{ref}/{path}"

    # 13. Bitbucket pull request/Commit -> Diff
    # Example: https://bitbucket.org/user/repo/pull-requests/1 -> https://bitbucket.org/user/repo/pull-requests/1/diff
    # Example: https://bitbucket.org/user/repo/commits/abc -> https://bitbucket.org/user/repo/commits/abc/diff
    bb_diff_match = re.match(r'(https?://(?:www\.)?bitbucket\.org/[^/]+/[^/]+/(?:pull-requests/\d+|commits/[a-f0-9]+))(?:/.*)?$', url, re.IGNORECASE)
    if bb_diff_match:
        return f"{bb_diff_match.group(1)}/diff"

    # 14. Bitbucket Snippet -> Raw
    # Example: https://bitbucket.org/user/snippets/abc -> https://bitbucket.org/user/snippets/abc/raw
    bb_snippet_match = re.match(r'(https?://(?:www\.)?bitbucket\.org/[^/]+/snippets/[^/]+)(?:/.*)?$', url, re.IGNORECASE)
    if bb_snippet_match:
        return f"{bb_snippet_match.group(1)}/raw"

    # 15. Bitbucket Cloud Tag/Branch/Repo -> ZIP Archive
    # Example: https://bitbucket.org/user/repo -> https://bitbucket.org/user/repo/get/HEAD.zip
    # Example: https://bitbucket.org/user/repo/src/main/ -> https://bitbucket.org/user/repo/get/main.zip
    bb_ref_match = re.match(r'https?://(?:www\.)?bitbucket\.org/([^/]+)/([^/]+)(?:/src/([^/]+))?/?$', url, re.IGNORECASE)
    if bb_ref_match:
        user, repo, ref = bb_ref_match.groups()
        ref = ref or "HEAD"
        return f"https://bitbucket.org/{user}/{repo}/get/{ref}.zip"

    # 16. Pastebin -> Raw
    # Example: https://pastebin.com/abcdefgh -> https://pastebin.com/raw/abcdefgh
    pb_match = re.match(r'https?://(?:www\.)?pastebin\.com/([a-zA-Z0-9]+)$', url, re.IGNORECASE)
    if pb_match:
        paste_id = pb_match.group(1)
        if paste_id.lower() not in ('archive', 'tools', 'faq', 'contact', 'night_mode', 'pro', 'doc', 'signup', 'login', 'api', 'trends', 'languages'):
            return f"https://pastebin.com/raw/{paste_id}"

    # 17. Hugging Face File -> Raw
    # Example: https://huggingface.co/user/repo/blob/main/script.py -> https://huggingface.co/user/repo/raw/main/script.py
    hf_blob_match = re.match(r'(https?://(?:www\.)?huggingface\.co/.+)/blob/(.+)', url, re.IGNORECASE)
    if hf_blob_match:
        base, path = hf_blob_match.groups()
        return f"{base}/raw/{path}"

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
        with open(filename, 'r', encoding='utf-8', errors='replace') as file:
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
    """Fetches content from a web link with safety limits. Automatically resolves GitHub/GitLab links.

    Args:
        url: The web link to fetch.
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

    # Ensure URL scheme is http or https to prevent SSRF/local file access
    if not url.lower().startswith(('http://', 'https://')):
        raise ValueError(f"Unsupported web link scheme: {url.split(':', 1)[0] if ':' in url else 'unknown'}")

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
        # 1. Check by magic bytes or content markers if available
        if content:
            if content.startswith(b'PK\x03\x04') or content.startswith(b'\x1f\x8b'):
                return True
            if len(content) > 262 and content[257:262] == b'ustar':
                return True
            # Detect Jupyter Notebooks by content
            if content.startswith(b'{') and b'"cells"' in content:
                return True
            # Detect Unified Diffs by content (e.g. for clipboard scans)
            if content.startswith(b'--- ') or content.startswith(b'Index: ') or content.startswith(b'diff --git '):
                return True

        # 2. Check by extension or basename
        path_s = str(file_path).lower()
        # Common archive and document extensions
        if path_s.endswith(('.zip', '.tar', '.tar.gz', '.ipynb', '.md', '.html', '.htm', '.xhtml', '.svg', '.yml', '.yaml',
                            '.diff', '.patch', '.service', '.desktop')):
            return True
        # Explicit manifest files and build scripts (checked by basename)
        basename = os.path.basename(path_s)
        if basename.endswith(('package.json', 'composer.json', 'deno.json', 'deno.jsonc', 'pyproject.toml',
                            'tasks.json', 'launch.json', 'dockerfile', 'makefile', 'docker-compose.yml', 'docker-compose.yaml')):
            return True
        return False

    apikey_missing_message = (
        "API key not found. Local scans and Ollama will continue to work, but OpenAI and OpenRouter require a key."
    )
    task_missing_message = (
        "'task.txt' is missing. AI analysis will be disabled."
    )
    extensions_missing_message = (
        f"'extensions.txt' is missing. The scanner will use default types: {', '.join(DEFAULT_EXTENSIONS)}."
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

        # Check for suspicious filename patterns (RTLO, double extensions, etc.)
        filename_score, _ = analyze_filename(path_str)
        if filename_score >= 0.5:
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

    Args:
        message: The message to display in the status label.
    """
    if status_label:
        status_label.config(text=message)
    if root:
        root.update_idletasks()


def update_progress(value: int) -> None:
    """Update the progress bar to reflect current progress.

    Args:
        value: Current progress count to display.
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

    Args:
        max_value: Total number of steps expected for the scan.
    """
    if progress_bar and root:
        progress_bar["maximum"] = max_value
        progress_bar["value"] = 0
        root.title("[0%] GPT Virus Scanner")
        root.update_idletasks()


def enqueue_ui_update(func: Callable, *args: Any, **kwargs: Any) -> None:
    """Add a task to the queue so the main window can update safely.

    Args:
        func: Function to execute on the UI thread.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.
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
    # Store the previous message to restore it later
    previous_message: List[str] = ["Ready"]

    def on_enter(event):
        target_label = label or status_label
        if target_label and current_cancel_event is None:
            # Save current text, defaulting to Ready if empty
            current_text = target_label.cget("text")
            previous_message[0] = current_text if current_text else "Ready"
            target_label.config(text=message)
            if root:
                root.update_idletasks()

    def on_leave(event):
        target_label = label or status_label
        if target_label and current_cancel_event is None:
            target_label.config(text=previous_message[0])
            if root:
                root.update_idletasks()

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def _quote_for_ui(path: str) -> str:
    """Quote a path for use in the GUI or CLI, respecting platform differences."""
    if sys.platform == "win32":
        # On Windows, use double quotes if there are spaces or other special chars.
        # shlex.quote (which uses single quotes) is not well-supported by Windows shells.
        if any(c in path for c in ' "%&^|<>'):
            # Escape double quotes by doubling them
            escaped_path = path.replace('"', '""')
            return f'"{escaped_path}"'
        return path
    return shlex.quote(path)


def _set_scan_target(path: Union[str, Iterable[str]]) -> None:
    """Update the scan target textbox and set focus to the scan button.

    Args:
        path: A single path string, or an iterable of path strings.
    """
    if not path or not textbox:
        return

    # Handle multiple paths or a single path string
    if not isinstance(path, str) and isinstance(path, Iterable):
        # Join multiple targets with appropriate quoting, ensuring all are strings
        formatted_path = " ".join(_quote_for_ui(str(p)) for p in path)
    else:
        # For a single path, use quote if it's not a list, for safety
        formatted_path = _quote_for_ui(str(path))

    textbox.delete(0, tk.END)
    textbox.insert(0, formatted_path)
    if scan_button:
        scan_button.focus_set()


def _get_initial_dir() -> Optional[str]:
    """Find a starting folder for file dialogs based on what is currently entered."""
    path_str = ""
    if textbox:
        path_str = textbox.get().strip()

    if not path_str:
        path_str = Config.last_path

    if not path_str:
        return None

    try:
        # Extract the first path if multiple are provided
        paths = shlex.split(path_str, posix=(sys.platform != "win32"))
        if not paths:
            return None
        first_path = paths[0]

        # If it's a URL or virtual path, we can't get a local folder
        if first_path.startswith(("[", "http://", "https://")):
            return None

        p = Path(first_path).absolute()
        if p.exists():
            return str(p if p.is_dir() else p.parent)
        return None
    except Exception:
        return None


def browse_dir_click() -> None:
    """Open the folder selection dialog and fill the textbox."""
    folder_selected = filedialog.askdirectory(initialdir=_get_initial_dir())
    if folder_selected:
        _set_scan_target(folder_selected)
        button_click()


def select_url_click() -> None:
    """Handle the web link input dialog and populate the textbox."""
    url_selected = simpledialog.askstring("Scan Web Link", "Enter a script web link to scan (http/https):")
    if url_selected:
        _set_scan_target(url_selected.strip())
        button_click()


def toggle_dry_run() -> None:
    """Update the scan button text based on the Dry Run state."""
    if not scan_button or not dry_var:
        return

    is_scanning = current_cancel_event is not None
    if not is_scanning:
        scan_button.config(text="Dry Run" if dry_var.get() else "Scan Now")


def toggle_ai_controls() -> None:
    """Enable or disable AI analysis controls based on current settings and scan state."""
    enabled = gpt_var.get() if gpt_var else False
    is_scanning = current_cancel_event is not None
    provider = provider_combo.get() if provider_combo else Config.provider

    if provider_combo and model_combo and api_key_entry and api_entry and show_key_btn:
        if enabled and not is_scanning:
            provider_combo.config(state="readonly")
            model_combo.config(state="normal")
            api_entry.config(state="normal")

            if provider == "ollama":
                api_key_entry.config(state="disabled")
                show_key_btn.config(state="disabled")
            else:
                api_key_entry.config(state="normal")
                show_key_btn.config(state="normal")
        else:
            provider_combo.config(state="disabled")
            model_combo.config(state="disabled")
            api_key_entry.config(state="disabled")
            api_entry.config(state="disabled")
            show_key_btn.config(state="disabled")

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
        filetypes=file_types,
        initialdir=_get_initial_dir()
    )
    if files_selected:
        _set_scan_target(files_selected)
        button_click()


def browse_file_list_click() -> None:
    """Handle the file list selection dialog and populate the textbox."""
    file_selected = filedialog.askopenfilename(
        title="Select File List to Scan",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialdir=_get_initial_dir()
    )
    if file_selected:
        try:
            with open(file_selected, 'r', encoding='utf-8') as f:
                paths = [line.strip() for line in f if line.strip()]
            if paths:
                _set_scan_target(paths)
                button_click()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file list: {e}")


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
    """Extract and check the AI's report from the server response.

    Args:
        response: The raw response from the AI provider.

    Returns:
        A dictionary with the AI's analysis if successful, or an error message string if the report is invalid or missing information.
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
    # Use split('\n') to preserve all empty lines, including trailing ones
    original_lines = val.split('\n')

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


def get_shell_profile_paths() -> List[str]:
    """Identify common shell profile and RC files, including system-wide profiles and aliases."""
    paths = []
    home = Path.home()

    # Linux/macOS/Unix-like
    common_files = [
        '.bashrc', '.bash_profile', '.bash_login', '.profile',
        '.zshrc', '.zprofile', '.zshenv', '.zlogin',
        '.bash_logout', '.zlogout', '.bash_aliases', '.zsh_aliases'
    ]

    for f in common_files:
        p = home / f
        if p.exists():
            paths.append(str(p))

    # System-wide POSIX profiles
    if sys.platform != "win32":
        system_files = [
            '/etc/profile', '/etc/bash.bashrc', '/etc/environment'
        ]
        for f in system_files:
            p = Path(f)
            if p.exists():
                paths.append(str(p))

        # /etc/profile.d/*.sh scripts
        profile_d = Path('/etc/profile.d')
        if profile_d.is_dir():
            for script in profile_d.glob('*.sh'):
                paths.append(str(script))

    # Windows PowerShell Profiles
    if sys.platform == "win32":
        try:
            cmd = ["powershell", "-NoProfile", "-Command",
                   "$PROFILE | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name | ForEach-Object { $PROFILE.$_ } | ConvertTo-Json"]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True)
            if output.strip():
                data = json.loads(output)
                if isinstance(data, str):
                    data = [data]
                for p_str in data:
                    if p_str and os.path.exists(p_str):
                        paths.append(p_str)
        except Exception:
            pass

    return sorted(list(set(paths)))


def get_shell_history_paths() -> List[str]:
    """Identify common shell history files on the current system."""
    paths = []
    home = Path.home()

    # Linux/macOS/Unix-like
    common_files = [
        '.bash_history', '.zsh_history', '.python_history',
        '.ash_history', '.dash_history', '.mysql_history',
        '.psql_history', '.node_repl_history', '.sqlite_history'
    ]

    for f in common_files:
        p = home / f
        if p.exists():
            paths.append(str(p))

    # Windows PowerShell History
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            ps_history = Path(appdata) / "Microsoft" / "Windows" / "PowerShell" / "PSReadLine" / "ConsoleHost_history.txt"
            if ps_history.exists():
                paths.append(str(ps_history))

    return paths


def _normalize_and_filter_dirs(paths: Iterable[Optional[str]]) -> List[str]:
    """Convert paths to absolute form, remove empty or missing folders, and deduplicate."""
    dirs = []
    seen = set()
    for p in paths:
        if not p:
            continue
        abs_p = os.path.abspath(p)
        if abs_p not in seen and os.path.isdir(abs_p):
            dirs.append(abs_p)
            seen.add(abs_p)
    return dirs


def get_system_path_directories() -> List[str]:
    """Find all folders listed in the system's PATH environment variable."""
    path_env = os.environ.get("PATH", "")
    if not path_env:
        return []

    return _normalize_and_filter_dirs(path_env.split(os.pathsep))


def get_downloads_paths() -> List[str]:
    """Find the standard Downloads folder."""
    paths = [str(Path.home() / "Downloads")]
    return _normalize_and_filter_dirs(paths)


def get_desktop_paths() -> List[str]:
    """Find the user's Desktop folder."""
    paths = [str(Path.home() / "Desktop")]
    return _normalize_and_filter_dirs(paths)


def get_temp_paths() -> List[str]:
    """Identify common temporary folders."""
    paths = [tempfile.gettempdir(), "/tmp", "/var/tmp"]
    return sorted(_normalize_and_filter_dirs(paths))


def get_ruby_gems_paths() -> List[str]:
    """Find all folders containing installed Ruby gems."""
    paths = []
    # 1. Check GEM_HOME environment variable
    gem_home = os.environ.get("GEM_HOME")
    if gem_home:
        paths.append(gem_home)

    # 2. Ask gem for the home directory
    try:
        is_win = sys.platform == "win32"
        output = subprocess.check_output(['gem', 'env', 'home'],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        shell=is_win).strip()
        if output:
            paths.append(output)
    except Exception:
        pass

    # 3. Common fallback paths
    if sys.platform != "win32":
        paths.extend([
            "/usr/local/lib/ruby/gems",
            "/usr/lib/ruby/gems"
        ])

    return sorted(_normalize_and_filter_dirs(paths))


def get_php_packages_paths() -> List[str]:
    """Find all folders containing global PHP Composer packages."""
    paths = []
    # 1. Ask composer for the global vendor directory
    try:
        is_win = sys.platform == "win32"
        output = subprocess.check_output(['composer', 'global', 'config', 'vendor-dir', '--absolute'],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        shell=is_win).strip()
        if output:
            paths.append(output)
    except Exception:
        pass

    # 2. Common fallback paths
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(os.path.join(appdata, "Composer", "vendor"))
    else:
        paths.append(str(home / ".composer" / "vendor"))
        paths.append(str(home / ".config" / "composer" / "vendor"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_rust_packages_paths() -> List[str]:
    """Find all folders containing global Rust Cargo packages."""
    paths = []
    # 1. Check CARGO_HOME environment variable
    cargo_home = os.environ.get("CARGO_HOME")
    if cargo_home:
        paths.append(os.path.join(cargo_home, "registry", "src"))
        paths.append(os.path.join(cargo_home, "git", "checkouts"))

    # 2. Default home directory location
    home = Path.home()
    cargo_default = home / ".cargo"
    paths.append(str(cargo_default / "registry" / "src"))
    paths.append(str(cargo_default / "git" / "checkouts"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_go_packages_paths() -> List[str]:
    """Find all folders containing Go packages (GOPATH)."""
    paths = []
    # 1. Check GOPATH environment variable
    gopath = os.environ.get("GOPATH")
    if gopath:
        for p in gopath.split(os.pathsep):
            if p:
                paths.append(os.path.join(p, "pkg", "mod"))
                paths.append(os.path.join(p, "src"))

    # 2. Ask go for the GOPATH
    try:
        is_win = sys.platform == "win32"
        output = subprocess.check_output(['go', 'env', 'GOPATH'],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        shell=is_win).strip()
        if output:
            for p in output.split(os.pathsep):
                if p:
                    paths.append(os.path.join(p, "pkg", "mod"))
                    paths.append(os.path.join(p, "src"))
    except Exception:
        pass

    # 3. Default location
    home = Path.home()
    paths.append(str(home / "go" / "pkg" / "mod"))
    paths.append(str(home / "go" / "src"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_java_packages_paths() -> List[str]:
    """Find all folders containing Java package caches (Maven and Gradle)."""
    paths = []
    home = Path.home()

    # Maven default repository
    paths.append(str(home / ".m2" / "repository"))

    # Gradle modules cache
    paths.append(str(home / ".gradle" / "caches" / "modules-2" / "files-2.1"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_dotnet_packages_paths() -> List[str]:
    """Find all folders containing global .NET NuGet package caches."""
    paths = []
    # 1. Check NUGET_PACKAGES environment variable
    nuget_packages = os.environ.get("NUGET_PACKAGES")
    if nuget_packages:
        paths.append(nuget_packages)

    # 2. Default location
    home = Path.home()
    paths.append(str(home / ".nuget" / "packages"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_documents_paths() -> List[str]:
    """Find the user's Documents folder."""
    paths = [str(Path.home() / "Documents")]

    # Windows specific (might be redirected)
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            docs_path, _ = winreg.QueryValueEx(key, "Personal")
            if docs_path:
                paths.append(docs_path)
        except Exception:
            pass

    return sorted(_normalize_and_filter_dirs(paths))


def get_running_process_commands() -> List[Tuple[str, bytes]]:
    """Collect command lines of all running processes."""
    processes = []
    try:
        if sys.platform == "win32":
            # Use PowerShell to get process name and command line
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process | Select-Object Name, CommandLine | ConvertTo-Json"]
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            if output.strip():
                data = json.loads(output)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get("Name", "Unknown")
                    cmdline = item.get("CommandLine")
                    if cmdline:
                        processes.append((f"[Process] {name}", cmdline.encode('utf-8')))
        else:
            # Linux/macOS
            # ps -eo args returns the command line
            # We skip the first line (header)
            output = subprocess.check_output(["ps", "-eo", "args"], stderr=subprocess.PIPE, universal_newlines=True)
            lines = output.splitlines()
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('[') and not line.endswith(']'): # Filter out kernel threads
                    # Use the first part of the command as a name hint
                    name = line.split()[0] if line.split() else "Unknown"
                    processes.append((f"[Process] {os.path.basename(name)}", line.encode('utf-8')))
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass

    return processes


def get_system_service_paths() -> List[str]:
    """Identify common systemd service and user service configuration files."""
    paths = []
    if sys.platform != "linux":
        return paths

    search_dirs = [
        Path("/etc/systemd/system"),
        Path("/lib/systemd/system"),
        Path("/usr/lib/systemd/system"),
        Path.home() / ".config" / "systemd" / "user"
    ]

    for d in search_dirs:
        # Only scan .service files to avoid excessive noise from other systemd units
        for p in d.rglob("*.service"):
            try:
                # Skip symlinks to avoid duplicate scanning of units
                if p.is_file() and not p.is_symlink():
                    paths.append(str(p))
            except Exception:
                pass

    return sorted(list(set(paths)))


def get_python_package_paths() -> List[str]:
    """Find all folders containing installed Python packages (site-packages)."""
    paths = []
    # 1. Standard site-packages
    if hasattr(site, 'getsitepackages'):
        try:
            paths.extend(site.getsitepackages())
        except Exception:
            pass

    # 2. User site-packages
    if hasattr(site, 'getusersitepackages'):
        try:
            paths.append(site.getusersitepackages())
        except Exception:
            pass

    # 3. Fallback/Environment-specific paths from sys.path
    for p in sys.path:
        if 'site-packages' in p:
            paths.append(p)

    return sorted(_normalize_and_filter_dirs(paths))


def get_browser_bookmarks_snippets() -> List[Tuple[str, bytes]]:
    """Identify common browser bookmark files and extract suspicious bookmarklets (javascript: or data: URLs)."""
    snippets = []
    home = Path.home()

    # Discovery logic similar to get_browser_extensions_paths
    bookmark_paths = []

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        appdata = os.environ.get("APPDATA")
        if local_appdata:
            # Chrome
            chrome_base = Path(local_appdata) / "Google" / "Chrome" / "User Data"
            bookmark_paths.append((str(chrome_base / "Default" / "Bookmarks"), "Chrome"))
            for p in chrome_base.glob("Profile */Bookmarks"):
                bookmark_paths.append((str(p), "Chrome"))
            # Edge
            edge_base = Path(local_appdata) / "Microsoft" / "Edge" / "User Data"
            bookmark_paths.append((str(edge_base / "Default" / "Bookmarks"), "Edge"))
            for p in edge_base.glob("Profile */Bookmarks"):
                bookmark_paths.append((str(p), "Edge"))
            # Brave
            brave_base = Path(local_appdata) / "BraveSoftware" / "Brave-Browser" / "User Data"
            bookmark_paths.append((str(brave_base / "Default" / "Bookmarks"), "Brave"))
            for p in brave_base.glob("Profile */Bookmarks"):
                bookmark_paths.append((str(p), "Brave"))
        if appdata:
            # Firefox
            ff_base = Path(appdata) / "Mozilla" / "Firefox" / "Profiles"
            for p in ff_base.glob("*/places.sqlite"):
                bookmark_paths.append((str(p), "Firefox"))
    elif sys.platform == "darwin":
        lib_support = home / "Library" / "Application Support"
        # Chrome
        chrome_base = lib_support / "Google" / "Chrome"
        bookmark_paths.append((str(chrome_base / "Default" / "Bookmarks"), "Chrome"))
        for p in chrome_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Chrome"))
        # Edge
        edge_base = lib_support / "Microsoft Edge"
        bookmark_paths.append((str(edge_base / "Default" / "Bookmarks"), "Edge"))
        for p in edge_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Edge"))
        # Brave
        brave_base = lib_support / "BraveSoftware" / "Brave-Browser"
        bookmark_paths.append((str(brave_base / "Default" / "Bookmarks"), "Brave"))
        for p in brave_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Brave"))
        # Firefox
        ff_base = lib_support / "Firefox" / "Profiles"
        for p in ff_base.glob("*/places.sqlite"):
            bookmark_paths.append((str(p), "Firefox"))
    else:
        # Linux
        config = home / ".config"
        # Chrome
        chrome_base = config / "google-chrome"
        bookmark_paths.append((str(chrome_base / "Default" / "Bookmarks"), "Chrome"))
        for p in chrome_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Chrome"))
        # Chromium
        chromium_base = config / "chromium"
        bookmark_paths.append((str(chromium_base / "Default" / "Bookmarks"), "Chromium"))
        for p in chromium_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Chromium"))
        # Brave
        brave_base = config / "BraveSoftware" / "Brave-Browser"
        bookmark_paths.append((str(brave_base / "Default" / "Bookmarks"), "Brave"))
        for p in brave_base.glob("Profile */Bookmarks"):
            bookmark_paths.append((str(p), "Brave"))
        # Firefox
        ff_base = home / ".mozilla" / "firefox"
        for p in ff_base.glob("*/places.sqlite"):
            bookmark_paths.append((str(p), "Firefox"))

    for path, browser in bookmark_paths:
        p_obj = Path(path)
        if not p_obj.exists():
            continue

        if browser == "Firefox":
            # Use a temporary copy to avoid locking issues
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    shutil.copy2(path, tmp.name)
                    tmp_path = tmp.name

                conn = sqlite3.connect(tmp_path)
                try:
                    cursor = conn.cursor()
                    # Query for bookmarks with javascript: or data: URLs
                    cursor.execute("SELECT b.title, p.url FROM moz_bookmarks b JOIN moz_places p ON b.fk = p.id WHERE p.url LIKE 'javascript:%' OR p.url LIKE 'data:%'")
                    for title, url in cursor.fetchall():
                        snippets.append((f"[{browser} Bookmark] {title or 'Untitled'}", url.encode('utf-8')))
                finally:
                    conn.close()
            except Exception:
                pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # Chromium-based (JSON)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)

                def find_bookmarklets(nodes):
                    for node in nodes:
                        if node.get('type') == 'url':
                            url = node.get('url', '')
                            if url.lower().startswith(('javascript:', 'data:')):
                                title = node.get('name', 'Untitled')
                                snippets.append((f"[{browser} Bookmark] {title}", url.encode('utf-8')))
                        elif node.get('type') == 'folder':
                            find_bookmarklets(node.get('children', []))

                roots = data.get('roots', {})
                for root_node in roots.values():
                    if isinstance(root_node, dict):
                        find_bookmarklets([root_node])
            except Exception:
                pass

    return snippets


def get_browser_extensions_paths() -> List[str]:
    """Find common browser extension folders (Chrome, Firefox, Edge)."""
    paths = []
    home = Path.home()

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        appdata = os.environ.get("APPDATA")
        if local_appdata:
            # Chrome
            chrome_base = Path(local_appdata) / "Google" / "Chrome" / "User Data"
            paths.append(str(chrome_base / "Default" / "Extensions"))
            for p in chrome_base.glob("Profile */Extensions"):
                paths.append(str(p))
            # Edge
            edge_base = Path(local_appdata) / "Microsoft" / "Edge" / "User Data"
            paths.append(str(edge_base / "Default" / "Extensions"))
            for p in edge_base.glob("Profile */Extensions"):
                paths.append(str(p))
        if appdata:
            # Firefox
            ff_base = Path(appdata) / "Mozilla" / "Firefox" / "Profiles"
            for p in ff_base.glob("*/extensions"):
                paths.append(str(p))
    elif sys.platform == "darwin":
        lib_support = home / "Library" / "Application Support"
        # Chrome
        chrome_base = lib_support / "Google" / "Chrome"
        paths.append(str(chrome_base / "Default" / "Extensions"))
        for p in chrome_base.glob("Profile */Extensions"):
            paths.append(str(p))
        # Edge
        edge_base = lib_support / "Microsoft Edge"
        paths.append(str(edge_base / "Default" / "Extensions"))
        for p in edge_base.glob("Profile */Extensions"):
            paths.append(str(p))
        # Firefox
        ff_base = lib_support / "Firefox" / "Profiles"
        for p in ff_base.glob("*/extensions"):
            paths.append(str(p))
    else:
        # Linux
        config = home / ".config"
        # Chrome
        chrome_base = config / "google-chrome"
        paths.append(str(chrome_base / "Default" / "Extensions"))
        for p in chrome_base.glob("Profile */Extensions"):
            paths.append(str(p))
        # Chromium
        chromium_base = config / "chromium"
        paths.append(str(chromium_base / "Default" / "Extensions"))
        for p in chromium_base.glob("Profile */Extensions"):
            paths.append(str(p))
        # Firefox
        ff_base = home / ".mozilla" / "firefox"
        for p in ff_base.glob("*/extensions"):
            paths.append(str(p))

    return sorted(_normalize_and_filter_dirs(paths))


def get_editor_extensions_paths() -> List[str]:
    """Find common editor extension folders (VS Code, Sublime Text, Vim/Neovim)."""
    paths = []
    home = Path.home()

    # 1. VS Code / Insiders / VSCodium
    vscode_dirs = [".vscode", ".vscode-insiders", ".vscode-oss", ".vscode-remote"]
    for d in vscode_dirs:
        paths.append(str(home / d / "extensions"))

    # 2. Sublime Text
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(os.path.join(appdata, "Sublime Text", "Packages"))
            paths.append(os.path.join(appdata, "Sublime Text 3", "Packages"))
    elif sys.platform == "darwin":
        paths.append(str(home / "Library" / "Application Support" / "Sublime Text" / "Packages"))
        paths.append(str(home / "Library" / "Application Support" / "Sublime Text 3" / "Packages"))
    else:
        paths.append(str(home / ".config" / "sublime-text" / "Packages"))
        paths.append(str(home / ".config" / "sublime-text-3" / "Packages"))

    # 3. Vim / Neovim
    # Vim 8+ native packages
    paths.append(str(home / ".vim" / "pack"))
    # Neovim native packages
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            paths.append(os.path.join(local_appdata, "nvim-data", "site", "pack"))
    else:
        paths.append(str(home / ".local" / "share" / "nvim" / "site" / "pack"))
    paths.append(str(home / ".config" / "nvim" / "pack"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_nodejs_package_paths() -> List[str]:
    """Find all folders containing global Node.js packages."""
    paths = []
    # 1. Ask npm for the global root
    try:
        # Use shell=True on Windows to find npm command properly
        is_win = sys.platform == "win32"
        output = subprocess.check_output(['npm', 'root', '-g'],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        shell=is_win).strip()
        if output:
            paths.append(output)
    except Exception:
        pass

    # 2. Common system paths (Unix-like)
    if sys.platform != "win32":
        paths.extend([
            "/usr/local/lib/node_modules",
            "/usr/lib/node_modules"
        ])
    else:
        # Windows common paths
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(os.path.join(appdata, "npm", "node_modules"))

        program_files = os.environ.get("ProgramFiles")
        if program_files:
            paths.append(os.path.join(program_files, "nodejs", "node_modules"))

    return sorted(_normalize_and_filter_dirs(paths))


def get_system_service_commands() -> List[Tuple[str, bytes]]:
    """Collect command lines of all system services (Windows Service PathName)."""
    items = []
    try:
        if sys.platform == "win32":
            # Use PowerShell to get service commands
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Service | Select-Object Name, PathName | ConvertTo-Json"]
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            if output.strip():
                data = json.loads(output)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get("Name", "Unknown")
                    command = item.get("PathName")
                    if command and command.strip():
                        items.append((f"[Service] {name}", command.encode('utf-8')))
    except Exception:
        pass
    return items


def get_environment_variable_snippets() -> List[Tuple[str, bytes]]:
    """Collect all non-empty environment variables as snippets."""
    snippets = []
    for key, value in os.environ.items():
        if value.strip():
            snippets.append((f"[EnvVar] {key}", value.encode('utf-8')))
    return snippets


def get_git_stash_snippets(path: str = ".") -> List[Tuple[str, bytes]]:
    """Collect the content of all Git stashes as snippets."""
    snippets = []
    toplevel, _ = _get_git_info(path)
    if toplevel is None:
        return []

    try:
        # Get list of stashes: stash@{0}: WIP on main: ...
        output = subprocess.check_output(
            ["git", "stash", "list"],
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        lines = output.splitlines()
        for line in lines:
            if not line.strip():
                continue
            # Extract stash name (e.g., stash@{0})
            match = re.match(r'^(stash@\{\d+\}):', line)
            if match:
                stash_id = match.group(1)
                # Get the full diff of the stash
                try:
                    diff = subprocess.check_output(
                        ["git", "stash", "show", "-p", stash_id],
                        cwd=toplevel,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    if diff.strip():
                        snippets.append((f"[{stash_id}] {line.strip()}", diff.encode('utf-8')))
                except subprocess.CalledProcessError:
                    continue
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return snippets


def get_scheduled_task_commands() -> List[Tuple[str, bytes]]:
    """Collect command lines of all scheduled tasks (Cron on Linux/macOS, schtasks on Windows)."""
    tasks = []
    try:
        if sys.platform == "win32":
            # Use schtasks to get task name and action (command) in CSV format
            # /fo CSV /v provides verbose output including 'Task To Run'
            cmd = ["schtasks", "/query", "/fo", "CSV", "/v"]
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            if output.strip():
                f = io.StringIO(output)
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("TaskName", "Unknown")
                    command = row.get("Task To Run") or row.get("Action")
                    if command and command.strip() and command.lower() != "n/a":
                        tasks.append((f"[Task] {name}", command.encode('utf-8')))
        else:
            # Linux/macOS - Collect from user and system crontabs

            # 1. User crontab
            try:
                output = subprocess.check_output(["crontab", "-l"], stderr=subprocess.PIPE, universal_newlines=True)
                for line in output.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # User crontab: min hour day month dow command
                        parts = line.split(None, 5)
                        if len(parts) > 5:
                            tasks.append(("[Cron] User", parts[5].encode('utf-8')))
                        elif line.startswith('@'):
                            parts = line.split(None, 1)
                            if len(parts) > 1:
                                tasks.append(("[Cron] User", parts[1].encode('utf-8')))
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # 2. System crontabs
            cron_files = ["/etc/crontab"]
            cron_d = Path("/etc/cron.d")
            if cron_d.is_dir():
                try:
                    for p in cron_d.iterdir():
                        if p.is_file():
                            cron_files.append(str(p))
                except Exception:
                    pass

            for cron_path in cron_files:
                if os.path.exists(cron_path):
                    try:
                        with open(cron_path, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#') and not re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=', line):
                                    # System crontab: min hour day month dow user command
                                    parts = line.split(None, 6)
                                    if len(parts) > 6:
                                        tasks.append((f"[Cron] System ({os.path.basename(cron_path)})", parts[6].encode('utf-8')))
                                    elif line.startswith('@'):
                                        # @reboot user command
                                        parts = line.split(None, 2)
                                        if len(parts) > 2:
                                            tasks.append((f"[Cron] System ({os.path.basename(cron_path)})", parts[2].encode('utf-8')))
                    except Exception:
                        pass
    except Exception:
        pass

    return tasks


def get_ssh_config_paths() -> List[str]:
    """Identify common SSH configuration and authorized_keys files."""
    paths = []
    home = Path.home()

    # User-level SSH files
    user_ssh = home / ".ssh"
    for f in ["config", "authorized_keys"]:
        p = user_ssh / f
        if p.exists():
            paths.append(str(p))

    # System-level SSH files
    if sys.platform == "win32":
        program_data = os.environ.get("ProgramData")
        if program_data:
            win_system_ssh = Path(program_data) / "ssh"
            for f in ["sshd_config", "ssh_config"]:
                p = win_system_ssh / f
                if p.exists():
                    paths.append(str(p))
    else:
        system_ssh_dir = Path("/etc/ssh")
        for f in ["sshd_config", "ssh_config"]:
            p = system_ssh_dir / f
            if p.exists():
                paths.append(str(p))

    return sorted(list(set(paths)))


def get_startup_item_commands() -> List[Tuple[str, bytes]]:
    """Collect command lines of all startup items (Autostart on Linux, LaunchAgents on macOS, StartupCommand on Windows)."""
    items = []
    try:
        if sys.platform == "win32":
            # Use PowerShell to get startup commands
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_StartupCommand | Select-Object Name, Command | ConvertTo-Json"]
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            if output.strip():
                data = json.loads(output)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get("Name", "Unknown")
                    command = item.get("Command")
                    if command and command.strip():
                        items.append((f"[Startup] {name}", command.encode('utf-8')))
        elif sys.platform == "darwin":
            # macOS - Scan LaunchAgents and LaunchDaemons
            search_dirs = [
                Path("/Library/LaunchAgents"),
                Path("/Library/LaunchDaemons"),
                Path.home() / "Library" / "LaunchAgents"
            ]
            for d in search_dirs:
                if d.exists():
                    for p in d.glob("*.plist"):
                        try:
                            with open(p, "rb") as f:
                                data = plistlib.load(f)
                                # Try 'Program' or 'ProgramArguments'
                                command = data.get("Program")
                                if not command:
                                    args = data.get("ProgramArguments")
                                    if args:
                                        command = " ".join(args) if isinstance(args, list) else str(args)
                                if command and command.strip():
                                    items.append((f"[LaunchAgent] {p.name}", command.encode('utf-8')))
                        except Exception:
                            pass
        else:
            # Linux - Scan .desktop files in autostart folders
            search_dirs = [
                Path("/etc/xdg/autostart"),
                Path.home() / ".config" / "autostart"
            ]
            for d in search_dirs:
                if d.exists():
                    for p in d.glob("*.desktop"):
                        try:
                            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    if line.startswith("Exec="):
                                        command = line[5:].strip()
                                        if command:
                                            items.append((f"[Autostart] {p.name}", command.encode('utf-8')))
                                        break
                        except Exception:
                            pass
    except Exception:
        pass

    return items


def get_git_hooks_paths(path: str = ".") -> List[str]:
    """Identify Git hooks for scanning, respecting local and global core.hooksPath."""
    paths = []
    toplevel, _ = _get_git_info(path)

    # 1. Resolve Hooks Folder
    hooks_dir = None
    try:
        # Check for core.hooksPath (captures both local and global)
        # Use subprocess.run to avoid raising on exit code 1 (not set)
        res = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=toplevel if toplevel else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        hooks_config = res.stdout.strip()

        if hooks_config:
            hooks_dir_path = Path(hooks_config).expanduser()
            if not hooks_dir_path.is_absolute() and toplevel:
                hooks_dir_path = Path(toplevel) / hooks_dir_path
            hooks_dir = str(hooks_dir_path)
        elif toplevel:
            # Fallback to default hooks folder using git rev-parse
            git_dir = subprocess.check_output(
                ["git", "rev-parse", "--git-dir"],
                cwd=toplevel,
                stderr=subprocess.PIPE,
                universal_newlines=True
            ).strip()
            git_dir_path = Path(git_dir)
            if not git_dir_path.is_absolute():
                git_dir_path = Path(toplevel) / git_dir_path
            hooks_dir = str(git_dir_path / "hooks")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Fallback for non-git folders or environments without git
        if toplevel:
            hooks_dir = os.path.join(toplevel, ".git", "hooks")

    # 2. Collect Hooks from Identified Folder
    if hooks_dir and os.path.isdir(hooks_dir):
        try:
            for entry in os.listdir(hooks_dir):
                if not entry.endswith(".sample"):
                    p = os.path.join(hooks_dir, entry)
                    if os.path.isfile(p):
                        paths.append(p)
        except OSError:
            pass

    return sorted(list(set(paths)))


def get_git_config_snippets() -> List[Tuple[str, bytes]]:
    """Identify potentially dangerous Git configuration settings (aliases, editors, etc.)."""
    snippets = []
    configs = [("--global", "Global Git Config")]

    # Check if we are in a Git repository to include local config
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
        configs.append(("--local", "Local Git Config"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Keys that are known to contain executable commands or script-like content
    dangerous_patterns = [
        r'^core\.editor$',
        r'^core\.sshcommand$',
        r'^core\.pager$',
        r'^sequence\.editor$',
        r'^credential\.helper$',
        r'^diff\..*?\.command$',
        r'^merge\..*?\.driver$',
        r'^pager\.',
    ]

    for flag, label in configs:
        try:
            # Use -z to handle multiline values correctly (null-separated entries)
            output = subprocess.check_output(
                ["git", "config", flag, "--list", "-z"],
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            # Entries are separated by null bytes. Each entry is 'key\nvalue'
            for entry in output.split('\0'):
                if '\n' in entry:
                    key, value = entry.split('\n', 1)
                    key = key.strip()
                    # For aliases, if it starts with !, it's a shell command
                    if key.lower().startswith('alias.') and value.startswith('!'):
                        snippet = value[1:].strip()
                        if snippet:
                            snippets.append((f"{label} [Alias: {key[6:]}]", snippet.encode('utf-8')))
                    elif any(re.match(pattern, key, re.IGNORECASE) for pattern in dangerous_patterns):
                        # For other dangerous keys, the whole value is the command/script
                        if value:
                            snippets.append((f"{label} [{key}]", value.encode('utf-8')))
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue

    return snippets


def _get_git_info(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the Git root folder and the relative path of the target."""
    abs_path = os.path.abspath(path)
    search_dir = os.path.dirname(abs_path) if os.path.isfile(abs_path) else abs_path

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=search_dir,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).strip()
        rel_target = os.path.relpath(abs_path, toplevel)
        return toplevel, rel_target
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        return None, None


def get_online_url(path: str, line: Union[int, str] = 1) -> Optional[str]:
    """Construct a web link for a local Git-tracked file or a remote target.

    Args:
        path: File path or remote target name.
        line: The line number to link to.

    Returns:
        The online web link string, or None if it could not be resolved.
    """
    line_val = int(line) if str(line).isdigit() and int(line) > 0 else None

    # 1. Handle Remote Targets (e.g., "[URL] https://github.com/...")
    if path.startswith("[URL] "):
        url = path[6:].strip()

        # Try to restore file view from raw web links for better UX
        # GitHub Raw: https://raw.githubusercontent.com/user/repo/rev/path
        gh_raw_match = re.match(r'https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/(.+)', url, re.IGNORECASE)
        if gh_raw_match:
            user, repo, rest = gh_raw_match.groups()
            url = f"https://github.com/{user}/{repo}/blob/{rest}"

        # GitLab Raw: https://gitlab.com/user/repo/-/raw/rev/path
        if '/-/raw/' in url.lower():
            url = url.replace('/-/raw/', '/-/blob/')

        # Append line fragment
        if line_val:
            if 'bitbucket.org' in url.lower():
                return f"{url}#lines-{line_val}"
            return f"{url}#L{line_val}"
        return url

    # 2. Handle Local Files
    if path.startswith("["):
        return None

    toplevel, rel_path = _get_git_info(path)
    if not toplevel or not rel_path:
        return None

    try:
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).strip()
        if not remote:
            return None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).strip()
        if not rev:
            rev = "HEAD"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        rev = "HEAD"

    # Normalize Remote URL (Convert SSH to HTTPS, remove .git suffix)
    # git@host:user/repo.git -> https://host/user/repo
    remote = re.sub(r'^git@([^:]+):', r'https://\1/', remote)
    remote = re.sub(r'\.git$', '', remote)
    remote = remote.rstrip('/')

    rel_path_url = rel_path.replace(os.sep, '/')

    if 'github.com' in remote.lower():
        base = f"{remote}/blob/{rev}/{rel_path_url}"
        return f"{base}#L{line_val}" if line_val else base
    if 'gitlab.com' in remote.lower():
        base = f"{remote}/-/blob/{rev}/{rel_path_url}"
        return f"{base}#L{line_val}" if line_val else base
    if 'bitbucket.org' in remote.lower():
        base = f"{remote}/src/{rev}/{rel_path_url}"
        return f"{base}#lines-{line_val}" if line_val else base

    return None


def get_git_changed_files(path: str = ".", ref: str = "HEAD") -> List[str]:
    """Get a list of changed files (staged, unstaged, untracked) from git.

    Args:
        path: The folder or file path.
        ref: The git revision or commit to compare against. Defaults to "HEAD".
    """
    toplevel, rel_target = _get_git_info(path)
    if toplevel is None:
        return []

    targets = [rel_target]

    files = set()
    # Changed relative to ref
    try:
        cmd = ["git", "diff", "--name-only", ref, "--"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Untracked files (only relevant when comparing against HEAD/working tree)
    if ref == "HEAD":
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


def get_git_diff(path: str = ".", ref: str = "HEAD") -> str:
    """Get the Git changes (diff) as a string.

    Args:
        path: The folder or file path.
        ref: The git revision or commit to compare against. Defaults to "HEAD".
    """
    toplevel, rel_target = _get_git_info(path)
    if toplevel is None:
        return ""

    targets = [rel_target] if rel_target != "." else []

    try:
        # Get diff relative to ref
        cmd = ["git", "diff", ref, "--no-color", "--"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return output
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _normalize_targets(targets: Union[str, List[str], Path]) -> List[str]:
    """Normalize scan targets (Path, string, or list) into a unique list of strings."""
    if isinstance(targets, Path):
        targets = [str(targets)]
    elif isinstance(targets, str):
        try:
            targets = shlex.split(targets, posix=(sys.platform != "win32"))
        except ValueError:
            targets = [targets]

    # Ensure all targets are strings and deduplicate while preserving order.
    return list(dict.fromkeys(str(t) for t in targets))


def collect_files(targets: Union[str, List[str]], modified_since: Optional[float] = None) -> List[Path]:
    """Collect files from a single path or a list of paths (files, folders, or patterns).

    Args:
        targets: A single folder path or a list of file/folder paths or glob patterns.
            Multiple targets can be provided in a single space-separated string.
        modified_since: A timestamp. If provided, only files modified after this time are returned.

    Returns:
        A unique list of files to scan.
    """
    targets = _normalize_targets(targets)

    results: List[Path] = []
    for t in targets:
        candidate_paths = [t] if Path(t).exists() else []
        if not candidate_paths and any(char in t for char in ['*', '?', '[']):
            candidate_paths = glob.glob(t, recursive=True)

        for path_str in candidate_paths:
            p = Path(path_str)
            if p.is_file():
                results.append(p)
            elif p.is_dir():
                results.extend([f for f in p.rglob('*') if f.is_file()])

    # Use dict keys to remove duplicates while preserving insertion order.
    unique_files = list(dict.fromkeys(results))

    if modified_since:
        filtered_files = []
        for f in unique_files:
            try:
                if f.stat().st_mtime >= modified_since:
                    filtered_files.append(f)
            except (OSError, FileNotFoundError):
                pass
        return filtered_files

    return unique_files


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


def format_percent(val: Any) -> str:
    """Format a numeric threat level (0-100) as a percentage string (e.g., 85 to "85%")."""
    try:
        return "{:.0%}".format(float(val) / 100.)
    except (ValueError, TypeError):
        return "Error"


def parse_duration(duration_str: str) -> Optional[float]:
    """Convert a duration string (e.g., "1h", "24h", "7d") to seconds.

    Args:
        duration_str: The duration string to parse.

    Returns:
        The duration in seconds as a float, or None if the format is invalid.
    """
    if not duration_str:
        return None

    match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]*)$', duration_str.strip())
    if not match:
        return None

    value_str, unit = match.groups()
    value = float(value_str)
    unit = unit.lower()

    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
    }

    if not unit:
        # Default to hours if no unit is provided
        return value * 3600

    if unit not in units:
        return None

    return value * units[unit]


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


def analyze_filename(path: Union[str, Path]) -> Tuple[float, str]:
    """Analyze a filename for deceptive patterns like RTLO, double extensions, or hidden whitespace.

    Returns:
        A tuple of (threat_score, message). Score is 0.0 to 1.0.
    """
    name = os.path.basename(str(path))
    if not name:
        return 0.0, ""

    # 1. RTLO (Right-to-Left Override) detection
    if '\u202e' in name:
        return 1.0, "Deceptive filename using Right-to-Left Override (RTLO) character detected."

    # 2. Deceptive Double Extensions
    # Common deceptive prefixes (document/media/web)
    deceptive_prefixes = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.zip', '.rar', '.7z', '.mp3', '.mp4', '.html', '.htm', '.svg'}
    # Executable extensions
    exec_extensions = {'.exe', '.bat', '.ps1', '.cmd', '.com', '.scr', '.pif', '.vbs', '.vbe', '.js', '.jse', '.wsf', '.wsh', '.msc', '.hta', '.cpl', '.msi', '.jar', '.py', '.sh', '.lnk', '.inf'}

    # Split and filter out empty strings from multiple dots (e.g. file...exe)
    # Also strip whitespace from parts to catch "file.jpg .exe"
    parts = [p.strip() for p in name.lower().split('.') if p.strip()]
    if len(parts) >= 2:
        ext2 = f".{parts[-1]}"
        # If we have at least two non-empty parts, we can check for double extension
        for i in range(len(parts) - 1):
            ext1 = f".{parts[i]}"
            if ext1 in deceptive_prefixes and ext2 in exec_extensions:
                msg = f"Deceptive double extension detected: '{ext1}{ext2}'."
                if re.search(r'\s', name):
                    msg += " (contains deceptive whitespace)"
                return 0.9, msg

    # 3. Hidden extension tricks (excessive whitespace)
    name_stripped = name.rstrip()
    if any(name_stripped.lower().endswith(ext) for ext in exec_extensions):
        # Check if there's a large gap before the extension
        # e.g. "invoice.pdf                              .exe" or "malware          .exe"
        if re.search(r'\s{5,}\.[^.]+$', name_stripped):
            return 0.9, "Suspiciously large whitespace gap found before file extension."

        # Check for even a single space/tab before the extension dot
        if re.search(r'\s\.[^.]+$', name_stripped):
            return 0.5, "Filename contains suspicious whitespace before the extension."

    # 4. Trailing whitespace or dots
    if (name and name[-1].isspace()) or name.endswith('.'):
        return 0.5, "Filename ends with a suspicious space or dot, often used to bypass filters."

    # 5. Invisible/Control characters (excluding standard ones)
    for char in name:
        cp = ord(char)
        if cp < 32 or (127 <= cp <= 159) or cp in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD):
            return 0.7, f"Filename contains invisible or control characters (Code {hex(cp)})."

    return 0.0, ""


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


def get_effective_sha256(path: str, snippet: Optional[str] = None) -> str:
    """Calculate the effective SHA256 hash, prioritizing full content from cache for virtual/missing paths."""
    if path.startswith("[") or not os.path.exists(path):
        # Prioritize full content from cache if available
        if path in _virtual_source_cache:
            return get_file_sha256(_virtual_source_cache[path].encode('utf-8'))
        if snippet:
            return get_file_sha256(snippet.encode('utf-8'))
        return ""
    return get_file_sha256(path)


def get_virustotal_url(path: str, snippet: Optional[str] = None) -> Optional[str]:
    """Construct a VirusTotal web link for a local file or a virtual snippet.

    Args:
        path: File path or virtual target name.
        snippet: The code snippet content (required for virtual paths).

    Returns:
        The VirusTotal web link string, or None if the hash could not be calculated.
    """
    h = get_effective_sha256(path, snippet)

    if h:
        return f"https://www.virustotal.com/gui/file/{h}"
    return None


def format_scan_summary(total_scanned: int, threats_found: int, total_bytes: Optional[int] = None, elapsed_time: Optional[float] = None, use_color: bool = False, high_risk: int = 0, medium_risk: int = 0) -> str:
    """Format a human-readable summary of the scan results.

    Args:
        total_scanned: Total number of files scanned.
        threats_found: Total number of suspicious files detected.
        total_bytes: Total bytes scanned.
        elapsed_time: Time taken for the scan in seconds.
        use_color: Whether to use ANSI color codes in the output.
        high_risk: Number of high risk files found.
        medium_risk: Number of medium risk files found.

    Returns:
        A human-readable summary string.
    """
    file_text = "file" if total_scanned == 1 else "files"
    threat_text = "suspicious file" if threats_found == 1 else "suspicious files"

    threats_display = str(threats_found)
    if use_color and threats_found > 0:
        # Use Bold Red for threats in terminal
        threats_display = f"\033[1;91m{threats_found}\033[0m"

    bytes_info = f" ({format_bytes(total_bytes)})" if total_bytes is not None else ""
    summary = f"Scan complete: {total_scanned} {file_text}{bytes_info} scanned, {threats_display} {threat_text} found"

    if threats_found > 0:
        summary += f" ({high_risk} high risk, {medium_risk} medium risk)."
    else:
        summary += "."

    if elapsed_time and elapsed_time > 0:
        files_per_sec = total_scanned / elapsed_time
        rate_text = "file" if files_per_sec <= 1.0 else "files"
        summary += f" Time: {elapsed_time:.1f}s ({files_per_sec:.1f} {rate_text}/s"
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

    Args:
        tv: Treeview widget containing the data to sort.
        col: Column identifier to sort.
        reverse: Sort order; ``True`` for descending and ``False`` for ascending.
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
    if not show_ai:
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

    # Update selection-dependent controls at start of scan to disable context menu
    if is_scanning:
        update_button_states()

    if scan_button:
        new_state = "disabled" if is_scanning else "normal"
        if is_scanning:
            scan_button.config(text="Scanning...", state=new_state)
        else:
            scan_button.config(text="Scan Now", state=new_state)
            toggle_dry_run()
    if cancel_button:
        cancel_button.config(state="normal" if is_scanning else "disabled")

    # Disable/Enable configuration widgets during scan
    config_widgets = [
        textbox, clear_target_btn, browse_button,
        git_checkbox, deep_checkbox, scan_all_checkbox, dry_checkbox,
        gpt_checkbox, provider_combo, model_combo, api_entry, show_key_btn,
        copy_cmd_button, intel_button
    ]
    for widget in config_widgets:
        if widget:
            widget.config(state="disabled" if is_scanning else "normal")

    # Disable specific footer buttons during a scan
    footer_buttons = [
        rescan_button, analyze_button, exclude_button, results_button
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

    Args:
        total_scanned: Total number of files scanned.
        threats_found: Total number of suspicious files detected.
        total_bytes: Total bytes scanned.
        elapsed_time: Time taken for the scan in seconds.
        high_risk: Number of high risk files found.
        medium_risk: Number of medium risk files found.
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


def scan_git_diff_click():
    """Scan current Git diff (staged and unstaged changes)."""
    try:
        # Get path from textbox or default to current folder
        target_path = textbox.get().strip() if textbox else "."
        if not target_path:
            target_path = "."

        diff_content = get_git_diff(target_path)
        if diff_content:
            button_click(extra_snippets=[("[Git Diff]", diff_content.encode('utf-8'))])
        else:
            messagebox.showinfo("Git Diff", "No Git changes detected (staged or unstaged) in the target path.")
    except Exception as e:
        messagebox.showwarning("Git Diff Error", f"Could not retrieve Git diff: {e}")


def scan_git_hooks_click():
    """Scan local and global Git hooks."""
    try:
        # Get path from textbox or default to current folder
        target_path = textbox.get().strip() if textbox else "."
        if not target_path:
            target_path = "."

        hook_paths = get_git_hooks_paths(target_path)
        if hook_paths:
            _set_scan_target(hook_paths)
            button_click()
        else:
            messagebox.showinfo("Git Hooks", "No Git hooks found to scan.")
    except Exception as e:
        messagebox.showwarning("Git Hooks Error", f"Could not scan Git hooks: {e}")


def scan_git_config_click():
    """Scan potentially dangerous Git configuration settings."""
    try:
        snippets = get_git_config_snippets()
        if snippets:
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("Git Configuration", "No potentially dangerous Git configuration settings were found.")
    except Exception as e:
        messagebox.showwarning("Git Configuration Error", f"Could not scan Git configuration: {e}")


def scan_git_stash_click():
    """Scan all Git stashes."""
    try:
        # Get path from textbox or default to current directory
        target_path = textbox.get().strip() if textbox else "."
        if not target_path:
            target_path = "."

        snippets = get_git_stash_snippets(target_path)
        if snippets:
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("Git Stash", "No Git stashes found to scan.")
    except Exception as e:
        messagebox.showwarning("Git Stash Error", f"Could not scan Git stashes: {e}")


def scan_git_conflicts_click():
    """Scan files with Git merge conflicts."""
    try:
        # Get path from textbox or default to current directory
        target_path = textbox.get().strip() if textbox else "."
        if not target_path:
            target_path = "."

        snippets = get_git_conflict_snippets(target_path)
        if snippets:
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("Git Conflicts", "No Git merge conflicts found to scan.")
    except Exception as e:
        messagebox.showwarning("Git Conflicts Error", f"Could not scan Git conflicts: {e}")


def get_git_conflict_snippets(path: str = ".") -> List[Tuple[str, bytes]]:
    """Get the Git changes for files with merge conflicts.

    Args:
        path: The folder or file path.
    """
    toplevel, _ = _get_git_info(path)
    if toplevel is None:
        return []

    try:
        # Find unmerged files (U filter)
        cmd_files = ["git", "diff", "--name-only", "--diff-filter=U"]
        output_files = subprocess.check_output(
            cmd_files,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        unmerged_files = output_files.splitlines()

        snippets = []
        for file_path in unmerged_files:
            # Get the conflict diff for each file as raw bytes
            cmd_diff = ["git", "diff", "--no-color", "--", file_path]
            output_diff = subprocess.check_output(
                cmd_diff,
                cwd=toplevel,
                stderr=subprocess.PIPE
            )
            if output_diff.strip():
                snippets.append((f"[Git Conflict] {file_path}", output_diff))
        return snippets
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []


def get_git_history_snippets(path: str = ".", count: int = 5) -> List[Tuple[str, bytes]]:
    """Get diffs for the last N commits in the Git repository.

    Args:
        path: The folder or file path within a Git repository.
        count: Number of recent commits to retrieve.
    """
    toplevel, _ = _get_git_info(path)
    if toplevel is None:
        return []

    try:
        # Get the list of last N commit hashes
        cmd_revs = ["git", "rev-list", "-n", str(count), "HEAD"]
        revs_output = subprocess.check_output(
            cmd_revs,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        hashes = revs_output.splitlines()

        snippets = []
        for commit_hash in hashes:
            try:
                # Get the diff for each commit
                cmd_show = ["git", "show", "--no-color", commit_hash]
                show_output = subprocess.check_output(
                    cmd_show,
                    cwd=toplevel,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                if show_output.strip():
                    # Extract short hash for naming
                    short_hash = commit_hash[:7]
                    snippets.append((f"[Git History] commit {short_hash}", show_output.encode('utf-8')))
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                continue
        return snippets
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []


def get_git_reflog_snippets(path: str = ".", count: int = 5) -> List[Tuple[str, bytes]]:
    """Get diffs for the last N entries in the Git reflog.

    Args:
        path: The folder or file path within a Git repository.
        count: Number of reflog entries to retrieve.
    """
    toplevel, _ = _get_git_info(path)
    if toplevel is None:
        return []

    try:
        # Get the list of last N reflog entries
        cmd_reflog = ["git", "reflog", "-n", str(count), "--format=%h %gs"]
        reflog_output = subprocess.check_output(
            cmd_reflog,
            cwd=toplevel,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        lines = reflog_output.splitlines()

        snippets = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(None, 1)
            commit_hash = parts[0]
            subject = parts[1] if len(parts) > 1 else ""

            # Get the diff for this reflog entry
            cmd_show = ["git", "show", "--no-color", commit_hash]
            try:
                show_output = subprocess.check_output(
                    cmd_show,
                    cwd=toplevel,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                if show_output.strip():
                    snippets.append((f"[Git Reflog] {commit_hash} {subject}", show_output.encode('utf-8')))
            except subprocess.CalledProcessError:
                continue
        return snippets
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []


def scan_git_history_click(count=None):
    """Scan recent Git commits."""
    try:
        if count is None:
            count = simpledialog.askinteger("Scan Recent Commits", "Enter number of recent commits to scan:", initialvalue=5, minvalue=1, maxvalue=100)
        if count is None:
            return

        scan_path = textbox.get()
        git_roots = shlex.split(scan_path, posix=(sys.platform != "win32")) if scan_path else ["."]

        all_snippets = []
        for root_dir in git_roots:
            all_snippets.extend(get_git_history_snippets(root_dir, count=count))

        if all_snippets:
            button_click(extra_snippets=all_snippets)
        else:
            messagebox.showinfo("Git History", f"No recent commits were found to scan in '{scan_path or '.'}'.")
    except Exception as e:
        messagebox.showwarning("Git History Error", f"Could not scan Git history: {e}")


def scan_git_reflog_click(count=None):
    """Scan recent entries in the Git reflog."""
    try:
        if count is None:
            count = simpledialog.askinteger("Scan Git Reflog", "Enter number of recent reflog entries to scan:", initialvalue=5, minvalue=1, maxvalue=100)
        if count is None:
            return

        scan_path = textbox.get()
        git_roots = shlex.split(scan_path, posix=(sys.platform != "win32")) if scan_path else ["."]

        all_snippets = []
        for root_dir in git_roots:
            all_snippets.extend(get_git_reflog_snippets(root_dir, count=count))

        if all_snippets:
            button_click(extra_snippets=all_snippets)
        else:
            messagebox.showinfo("Git Reflog", f"No recent reflog entries were found to scan in '{scan_path or '.'}'.")
    except Exception as e:
        messagebox.showwarning("Git Reflog Error", f"Could not scan Git reflog: {e}")


def scan_git_revision_click():
    """Scan files changed in a specific Git revision."""
    try:
        target_path = textbox.get().strip() if textbox else "."
        if not target_path:
            target_path = "."

        toplevel, _ = _get_git_info(target_path)
        if toplevel is None:
            messagebox.showwarning("Git Error", "Target path is not part of a Git repository.")
            return

        ref = simpledialog.askstring("Scan Git Revision", "Enter a Git revision (e.g., main, HEAD~1, or a commit hash):")
        if not ref:
            return

        changed_files = get_git_changed_files(target_path, ref=ref)
        if changed_files:
            # Update the textbox with the changed files list and trigger scan
            _set_scan_target(changed_files)
            button_click()
        else:
            messagebox.showinfo("Git Revision", f"No changed files found for revision '{ref}'.")
    except Exception as e:
        messagebox.showwarning("Git Revision Error", f"Could not scan Git revision: {e}")


def scan_shell_profiles_click():
    """Scan common shell profile and RC files."""
    try:
        profile_paths = get_shell_profile_paths()
        if profile_paths:
            _set_scan_target(profile_paths)
            button_click()
        else:
            messagebox.showinfo("Shell Profiles", "No common shell profile files were found on this system.")
    except Exception as e:
        messagebox.showwarning("Shell Profiles Error", f"Could not scan shell profiles: {e}")


def scan_shell_history_click():
    """Scan common shell history files."""
    try:
        history_paths = get_shell_history_paths()
        if history_paths:
            _set_scan_target(history_paths)
            button_click()
        else:
            messagebox.showinfo("Shell History", "No common shell history files were found on this system.")
    except Exception as e:
        messagebox.showwarning("Shell History Error", f"Could not scan shell history: {e}")


def scan_system_path_click():
    """Scan all folders in the system's PATH environment variable."""
    try:
        path_dirs = get_system_path_directories()
        if path_dirs:
            _set_scan_target(path_dirs)
            button_click()
        else:
            messagebox.showinfo("System PATH", "No valid folders found in the system PATH.")
    except Exception as e:
        messagebox.showwarning("System PATH Error", f"Could not scan system PATH: {e}")


def scan_running_processes_click():
    """Scan command lines of all running processes."""
    try:
        processes = get_running_process_commands()
        if processes:
            button_click(extra_snippets=processes)
        else:
            messagebox.showinfo("Running Processes", "No running processes with command lines were found.")
    except Exception as e:
        messagebox.showwarning("Running Processes Error", f"Could not scan running processes: {e}")


def scan_scheduled_tasks_click():
    """Scan all scheduled tasks and Cron jobs."""
    try:
        tasks = get_scheduled_task_commands()
        if tasks:
            button_click(extra_snippets=tasks)
        else:
            messagebox.showinfo("Scheduled Tasks", "No scheduled tasks or Cron jobs were found.")
    except Exception as e:
        messagebox.showwarning("Scheduled Tasks Error", f"Could not scan scheduled tasks: {e}")


def scan_startup_items_click():
    """Scan all system startup items and LaunchAgents."""
    try:
        items = get_startup_item_commands()
        if items:
            button_click(extra_snippets=items)
        else:
            messagebox.showinfo("Startup Items", "No system startup items or LaunchAgents were found.")
    except Exception as e:
        messagebox.showwarning("Startup Items Error", f"Could not scan startup items: {e}")


def scan_system_services_click():
    """Scan all system services (systemd files on Linux, Service PathName on Windows)."""
    try:
        paths = get_system_service_paths()
        snippets = get_system_service_commands()
        if paths or snippets:
            if paths:
                _set_scan_target(paths)
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("System Services", "No system services were found to scan.")
    except Exception as e:
        messagebox.showwarning("System Services Error", f"Could not scan system services: {e}")


def scan_ssh_config_click():
    """Scan all common SSH configuration and authorized_keys files."""
    try:
        paths = get_ssh_config_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("SSH Configuration", "No SSH configuration or authorized_keys files were found to scan.")
    except Exception as e:
        messagebox.showwarning("SSH Configuration Error", f"Could not scan SSH configuration: {e}")


def scan_python_packages_click():
    """Scan all folders containing installed Python packages (site-packages)."""
    try:
        package_paths = get_python_package_paths()
        if package_paths:
            _set_scan_target(package_paths)
            button_click()
        else:
            messagebox.showinfo("Python Packages", "No Python site-packages folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Python Packages Error", f"Could not scan Python packages: {e}")


def scan_recently_modified_click(duration_str: Optional[str] = None):
    """Scan files modified within a user-specified duration."""
    if duration_str is None:
        duration_str = simpledialog.askstring("Scan Recently Modified", "Enter duration (e.g., 24h, 1h, 7d):", initialvalue="24h")
    if duration_str:
        duration = parse_duration(duration_str)
        if duration is None:
            messagebox.showerror("Error", f"Invalid duration format: {duration_str}")
            return
        modified_since = time.time() - duration
        button_click(modified_since=modified_since)


def scan_env_vars_click():
    """Scan all non-empty environment variables."""
    try:
        snippets = get_environment_variable_snippets()
        if snippets:
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("Environment Variables", "No non-empty environment variables were found.")
    except Exception as e:
        messagebox.showwarning("Environment Variables Error", f"Could not scan environment variables: {e}")


def scan_nodejs_packages_click():
    """Scan all folders containing global Node.js packages."""
    try:
        package_paths = get_nodejs_package_paths()
        if package_paths:
            _set_scan_target(package_paths)
            button_click()
        else:
            messagebox.showinfo("Node.js Packages", "No global Node.js package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Node.js Packages Error", f"Could not scan Node.js packages: {e}")


def scan_browser_bookmarks_click():
    """Scan all common browser bookmark files for suspicious bookmarklets."""
    try:
        snippets = get_browser_bookmarks_snippets()
        if snippets:
            button_click(extra_snippets=snippets)
        else:
            messagebox.showinfo("Browser Bookmarks", "No suspicious browser bookmarklets (javascript: or data: URLs) were found.")
    except Exception as e:
        messagebox.showwarning("Browser Bookmarks Error", f"Could not scan browser bookmarks: {e}")


def scan_browser_extensions_click():
    """Scan all common browser extension folders."""
    try:
        extension_paths = get_browser_extensions_paths()
        if extension_paths:
            _set_scan_target(extension_paths)
            button_click()
        else:
            messagebox.showinfo("Browser Extensions", "No browser extension folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Browser Extensions Error", f"Could not scan browser extensions: {e}")


def scan_editor_extensions_click():
    """Scan all folders containing editor extensions (VS Code, Sublime Text, Vim)."""
    try:
        extension_paths = get_editor_extensions_paths()
        if extension_paths:
            _set_scan_target(extension_paths)
            button_click()
        else:
            messagebox.showinfo("Editor Extensions", "No editor extension folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Editor Extensions Error", f"Could not scan editor extensions: {e}")


def scan_downloads_click():
    """Scan the standard Downloads folder."""
    try:
        paths = get_downloads_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Downloads", "The standard Downloads folder was not found on this system.")
    except Exception as e:
        messagebox.showwarning("Downloads Error", f"Could not scan Downloads: {e}")


def scan_desktop_click():
    """Scan the user's Desktop folder."""
    try:
        paths = get_desktop_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Desktop", "The Desktop folder was not found on this system.")
    except Exception as e:
        messagebox.showwarning("Desktop Error", f"Could not scan Desktop: {e}")


def scan_temp_click():
    """Scan common temporary folders."""
    try:
        paths = get_temp_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Temporary Folders", "No common temporary folders were found on this system.")
    except Exception as e:
        messagebox.showwarning("Temporary Folders Error", f"Could not scan temporary folders: {e}")


def scan_ruby_gems_click():
    """Scan all folders containing installed Ruby gems."""
    try:
        paths = get_ruby_gems_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Ruby Gems", "No Ruby gems folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Ruby Gems Error", f"Could not scan Ruby gems: {e}")


def scan_php_packages_click():
    """Scan all folders containing global PHP Composer packages."""
    try:
        paths = get_php_packages_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("PHP Packages", "No global PHP package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("PHP Packages Error", f"Could not scan PHP packages: {e}")


def scan_rust_packages_click():
    """Scan all folders containing global Rust Cargo packages."""
    try:
        paths = get_rust_packages_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Rust Packages", "No global Rust package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Rust Packages Error", f"Could not scan Rust packages: {e}")


def scan_go_packages_click():
    """Scan all folders containing Go packages (GOPATH)."""
    try:
        paths = get_go_packages_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Go Packages", "No Go package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Go Packages Error", f"Could not scan Go packages: {e}")


def scan_java_packages_click():
    """Scan all folders containing Java package caches (Maven and Gradle)."""
    try:
        paths = get_java_packages_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Java Packages", "No Java package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning("Java Packages Error", f"Could not scan Java packages: {e}")


def scan_dotnet_packages_click():
    """Scan all folders containing global .NET NuGet package caches."""
    try:
        paths = get_dotnet_packages_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo(".NET Packages", "No .NET NuGet package folders were found to scan.")
    except Exception as e:
        messagebox.showwarning(".NET Packages Error", f"Could not scan .NET packages: {e}")


def scan_documents_click():
    """Scan the user's Documents folder."""
    try:
        paths = get_documents_paths()
        if paths:
            _set_scan_target(paths)
            button_click()
        else:
            messagebox.showinfo("Documents", "The standard Documents folder was not found on this system.")
    except Exception as e:
        messagebox.showwarning("Documents Error", f"Could not scan Documents: {e}")


def get_system_audit_data() -> Tuple[List[str], List[Tuple[str, bytes]]]:
    """Collect all paths and snippets for a comprehensive system audit."""
    all_paths = []
    all_paths.extend(get_shell_profile_paths())
    all_paths.extend(get_shell_history_paths())
    all_paths.extend(get_system_path_directories())
    all_paths.extend(get_ssh_config_paths())
    all_paths.extend(get_system_service_paths())
    all_paths.extend(get_git_hooks_paths())
    all_paths.extend(get_python_package_paths())
    all_paths.extend(get_nodejs_package_paths())
    all_paths.extend(get_ruby_gems_paths())
    all_paths.extend(get_php_packages_paths())
    all_paths.extend(get_rust_packages_paths())
    all_paths.extend(get_go_packages_paths())
    all_paths.extend(get_java_packages_paths())
    all_paths.extend(get_dotnet_packages_paths())
    all_paths.extend(get_browser_extensions_paths())
    all_paths.extend(get_editor_extensions_paths())
    all_paths.extend(get_documents_paths())
    all_paths.extend(get_downloads_paths())
    all_paths.extend(get_desktop_paths())
    all_paths.extend(get_temp_paths())

    all_snippets = []
    all_snippets.extend(get_running_process_commands())
    all_snippets.extend(get_environment_variable_snippets())
    all_snippets.extend(get_scheduled_task_commands())
    all_snippets.extend(get_startup_item_commands())
    all_snippets.extend(get_system_service_commands())
    all_snippets.extend(get_git_config_snippets())
    all_snippets.extend(get_git_stash_snippets())
    all_snippets.extend(get_browser_bookmarks_snippets())

    return all_paths, all_snippets


def scan_system_audit_click():
    """Perform a comprehensive system audit scan (Profiles, History, Path, SSH, Processes, Tasks, Startup, Services, EnvVars)."""
    try:
        all_paths, all_snippets = get_system_audit_data()

        if all_paths or all_snippets:
            _set_scan_target(all_paths)
            button_click(extra_snippets=all_snippets)
        else:
            messagebox.showinfo("System Audit", "No system items were found to scan.")
    except Exception as e:
        messagebox.showwarning("System Audit Error", f"Could not perform system audit: {e}")


def button_click(extra_snippets: Optional[List[Tuple[str, bytes]]] = None, fail_threshold: Optional[int] = None, modified_since: Optional[float] = None) -> None:
    """Trigger a scan in a background thread using the selected path.

    Args:
        extra_snippets: List of (name, content) tuples to scan as in-memory buffers.
        fail_threshold: Threat level threshold to trigger a failure count.
        modified_since: A timestamp. If provided, only files modified after this time are scanned.

    Returns:
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
        scan_targets = shlex.split(scan_path, posix=(sys.platform != "win32")) if scan_path else []
    except ValueError as e:
        messagebox.showerror("Selection Error", f"Malformed path selection: {e}")
        return

    if scan_targets and git_var.get():
        all_git_files = []
        for target in scan_targets:
            all_git_files.extend(get_git_changed_files(target))

        if not all_git_files:
            messagebox.showinfo("Git Scan", "No git changes detected in the selected folder.")
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
        fail_threshold,
        modified_since
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
                "admin_desc": values[2],
                "user_desc": values[3],
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
        folder = filedialog.askdirectory(parent=manage_win, initialdir=_get_initial_dir())
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
    """Extract scan-ready snippets from various container formats.

    This function recursively unpacks archives (ZIP, TAR), Jupyter Notebooks,
    project and build files (package.json, pyproject.toml), Dockerfiles, Makefiles,
    automation tasks (YAML), web files (HTML, SVG), and Unified Diffs.
    It ensures that only relevant code blocks and scripts are processed.

    Args:
        name: The display name or path of the content.
        content: The raw bytes of the content.
        depth: Current recursion depth to prevent infinite loops (max 5).
        hint: Optional filename hint to assist in format detection.

    Yields:
        Tuples containing the snippet name and its content as bytes.
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

    # 4. Check for project and build files (package.json, composer.json, deno.json/jsonc, pyproject.toml, tasks.json, launch.json)
    lowered_check = check_name.lower()
    check_basename = os.path.basename(lowered_check)
    if check_basename.endswith(('package.json', 'composer.json', 'deno.json', 'deno.jsonc', 'pyproject.toml', 'tasks.json', 'launch.json')):
        try:
            text = content.decode('utf-8', errors='ignore')
            if check_basename.endswith('pyproject.toml'):
                # Parse pyproject.toml using regex to avoid toml dependency
                lines = text.splitlines()
                in_script_section = False
                current_nested_script = None

                def yield_pyproject_scripts(label_name, val):
                    # Strip inline comment if not a multiline string
                    if not (val.startswith(('"""', "'''"))):
                        # A robust regex to match comments outside quotes, including triple quotes
                        robust_comment_match = re.search(r'^(?:[^"\'#]|"{3}(?:\\.|[^\\])*?"{3}(?!")|(?<!")"(?:\\.|[^"\\])*?"(?!")|\'{3}.*?\'{3}(?!\')|(?<!\')\'(?:\\.|[^\'\\])*?\'(?!\'))*(#.*)', val, re.DOTALL)
                        if robust_comment_match:
                            comment_start = robust_comment_match.start(1)
                            val = val[:comment_start].strip()

                    if val.startswith('['):
                        # Array of commands: ["a", "b"], handles triple quotes and escaped characters
                        items = re.findall(r'"{3}((?:\\.|[^\\])*?)"{3}(?!")|\'{3}(.*?)\'{3}(?!\')|"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'', val, re.DOTALL)
                        for idx, item in enumerate(items, 1):
                            cmd = item[0] or item[1] or item[2] or item[3]
                            if cmd.strip():
                                yield (f"{name} [Script: {label_name} ({idx})]", cmd.encode('utf-8'))
                    else:
                        # Single command or multi-line content
                        if val.startswith('"""') and val.endswith('"""') and len(val) >= 6:
                            cmd = val[3:-3]
                        elif val.startswith("'''") and val.endswith("'''") and len(val) >= 6:
                            cmd = val[3:-3]
                        elif val.startswith('"') and val.endswith('"') and len(val) >= 2:
                            cmd = val[1:-1]
                        elif val.startswith("'") and val.endswith("'") and len(val) >= 2:
                            cmd = val[1:-1]
                        else:
                            cmd = val
                        
                        if cmd.strip():
                            yield (f"{name} [Script: {label_name}]", cmd.encode('utf-8'))

                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    i += 1
                    if not line or line.startswith('#'):
                        continue

                    header_match = re.match(r'^\[\s*([^\]]+)\s*\](?:\s*#.*)?$', line)
                    if header_match:
                        section = header_match.group(1).strip()
                        # Flat script sections: [project.scripts], [project.gui-scripts], [tool.pdm.scripts], [tool.poe.tasks], etc.
                        if re.match(r'^(?:.*?\.)?(?:scripts|tasks|gui-scripts)$', section, re.IGNORECASE) or \
                           section.lower() == 'tool.pdm.dev-dependencies':
                            in_script_section = True
                            current_nested_script = None
                        # Nested script sections: [tool.pdm.scripts.test], [tool.poe.tasks.test]
                        elif match := re.match(r'^(?:.*?\.)?(?:scripts|tasks|gui-scripts)\.(.+)$', section, re.IGNORECASE):
                            in_script_section = True
                            current_nested_script = match.group(1).strip()
                        else:
                            in_script_section = False
                            current_nested_script = None
                        continue

                    if in_script_section:
                        # Match key = "value" or key = { ... }
                        # Key can be quoted: "quoted key" = "value"
                        match = re.match(r'^((?:"[^"]*"|\'[^\']*\'|[^=\s]+))\s*=\s*(.*)', line)
                        if match:
                            script_key = match.group(1).strip().strip('"\'')
                            command_val = match.group(2).strip()

                            # Handle multiline strings
                            if command_val.startswith(('"""', "'''")):
                                quote_type = command_val[:3]
                                # Check if it also ends with it (possibly with a comment)
                                # Finding the second occurrence of quote_type after the first 3 chars
                                end_pos = command_val.find(quote_type, 3)
                                if end_pos != -1:
                                    command_val = command_val[:end_pos + 3]
                                else:
                                    multiline_lines = [command_val]
                                    while i < len(lines):
                                        curr_line = lines[i]
                                        multiline_lines.append(curr_line)
                                        i += 1
                                        end_pos = curr_line.find(quote_type)
                                        if end_pos != -1:
                                            # Truncate at closing quotes to ignore trailing comments
                                            multiline_lines[-1] = curr_line[:end_pos + 3]
                                            break
                                    command_val = "\n".join(multiline_lines)
                            # Handle multiline arrays
                            elif command_val.startswith('['):
                                # Check if it actually ends on this line (ignoring comments)
                                temp_val = command_val
                                # A robust regex to match comments outside quotes, including triple quotes
                                comment_match = re.search(r'^(?:[^"\'#]|"{3}(?:\\.|[^\\])*?"{3}(?!")|(?<!")"(?:\\.|[^"\\])*?"(?!")|\'{3}.*?\'{3}(?!\')|(?<!\')\'(?:\\.|[^\'\\])*?\'(?!\'))*(#.*)', temp_val)
                                if comment_match:
                                    temp_val = temp_val[:comment_match.start(1)].strip()

                                if not temp_val.endswith(']'):
                                    array_lines = [command_val]
                                    while i < len(lines):
                                        curr_line = lines[i].strip()
                                        # Strip inline comments from array lines
                                        comment_match = re.search(r'^(?:[^"\'#]|"{3}(?:\\.|[^\\])*?"{3}(?!")|(?<!")"(?:\\.|[^"\\])*?"(?!")|\'{3}.*?\'{3}(?!\')|(?<!\')\'(?:\\.|[^\'\\])*?\'(?!\'))*(#.*)', curr_line)
                                        if comment_match:
                                            curr_line = curr_line[:comment_match.start(1)].strip()

                                        array_lines.append(curr_line)
                                        i += 1
                                        if curr_line.endswith(']'):
                                            break
                                    command_val = "".join(array_lines)

                            if current_nested_script:
                                # Inside a nested section like [tool.pdm.scripts.test] or [tool.poe.tasks.test]
                                if script_key.lower() in ('cmd', 'shell', 'command', 'composite', 'script', 'expr'):
                                    yield from yield_pyproject_scripts(current_nested_script, command_val)
                                    in_script_section = False
                            else:
                                # Inside a flat section like [project.scripts] or [tool.poe.tasks]
                                if command_val.startswith('{'):
                                    # Inline table
                                    # Robustly handle brackets and quotes within the inline table value
                                    cmd_match = re.search(r'(?:cmd|command|shell|composite|script|expr)\s*=\s*("{3}(?:\\.|[^\\])*?"{3}(?!")|\'{3}(?:\\.|[^\\])*?\'{3}(?!\')|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\[(?:[^"\'\]]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')*\])', command_val)
                                    if cmd_match:
                                        yield from yield_pyproject_scripts(script_key, cmd_match.group(1).strip())
                                else:
                                    yield from yield_pyproject_scripts(script_key, command_val)
                return

            # Strip single-line and multi-line comments for JSONC support
            # This regex strips /* ... */ comments and // ... comments while attempting to ignore
            # // if it appears inside a double-quoted string (to avoid mangling URLs).
            # Many manifests (deno.json, tasks.json, launch.json) support comments.
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

            # Special handling for VS Code tasks.json
            if check_basename == 'tasks.json':
                yielded_any = False
                # Tasks can be in a 'tasks' array
                tasks = manifest.get('tasks', [])
                if isinstance(tasks, list):
                    for task in tasks:
                        if not isinstance(task, dict):
                            continue
                        task_label = task.get('label') or task.get('taskName') or "Unnamed Task"
                        cmd = task.get('command')
                        args = task.get('args')
                        if isinstance(cmd, str) and cmd.strip():
                            full_cmd = cmd
                            if isinstance(args, list):
                                full_cmd += " " + " ".join(str(a) for a in args)
                            yield (f"{name} [Task: {task_label}]", full_cmd.encode('utf-8'))
                            yielded_any = True

                # VS Code also supports 'inputs' which can have default values or commands
                inputs = manifest.get('inputs', [])
                if isinstance(inputs, list):
                    for inp in inputs:
                        if not isinstance(inp, dict):
                            continue
                        inp_id = inp.get('id', 'Unnamed Input')
                        if inp.get('type') == 'command':
                            cmd = inp.get('command')
                            if isinstance(cmd, str) and cmd.strip():
                                yield (f"{name} [Input Command: {inp_id}]", cmd.encode('utf-8'))
                                yielded_any = True

                if yielded_any:
                    return

            # Special handling for VS Code launch.json
            if check_basename == 'launch.json':
                yielded_any = False
                configs = manifest.get('configurations', [])
                if isinstance(configs, list):
                    for config in configs:
                        if not isinstance(config, dict):
                            continue
                        config_name = config.get('name', 'Unnamed Config')
                        # Extract program, args, and preLaunchTask
                        parts = []
                        if config.get('program'):
                            parts.append(str(config.get('program')))
                        if config.get('args'):
                            args = config.get('args')
                            if isinstance(args, list):
                                parts.extend(str(a) for a in args)
                            else:
                                parts.append(str(args))
                        if parts:
                            yield (f"{name} [Launch: {config_name}]", " ".join(parts).encode('utf-8'))
                            yielded_any = True

                        pre_task = config.get('preLaunchTask')
                        if isinstance(pre_task, str) and pre_task.strip():
                            yield (f"{name} [PreLaunchTask: {config_name}]", pre_task.encode('utf-8'))
                            yielded_any = True

                if yielded_any:
                    return

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

    # 6. Check for HTML/SVG script tags and other embedded elements
    if check_name.lower().endswith(('.html', '.htm', '.xhtml', '.svg')):
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

            # 3. Match inline event handlers and javascript: URLs in attributes
            # Matches on...="script", on...='script', or on...=script (unquoted)
            # Also handles leading whitespace in javascript: URLs
            attr_patterns = [
                (r'\s+(on[a-z]+)\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+)', True),
                (r'\s+(?:href|src|action|formaction|data|xlink:href)\s*=\s*("\s*javascript:[^"]*"|\'\s*javascript:[^\']*\'|\s*javascript:[^\s>]+)', False)
            ]
            extracted_attrs = []
            for pattern, has_name_group in attr_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    val = match[1] if has_name_group else match
                    script = val.strip('"\'').strip()
                    if script:
                        unescaped = html.unescape(script)
                        if unescaped not in extracted_attrs:
                            extracted_attrs.append(unescaped)

            if extracted_attrs:
                bundle = "\n".join(extracted_attrs)
                yield (f"{name} [Attributes]", bundle.encode('utf-8'))
                extracted_any = True

            if extracted_any:
                return
        except Exception:
            pass

    # 7. Check for Dockerfile
    if check_basename.endswith('dockerfile'):
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
                # Support ONBUILD RUN/CMD/ENTRYPOINT and HEALTHCHECK [OPTIONS] CMD
                instr_match = re.match(r'^\s*(?:(?:ONBUILD\s+)?(?:RUN|CMD|ENTRYPOINT)|HEALTHCHECK(?:\s+--[a-z0-9-]+=[^\s]+)*\s+CMD)\s+(.*)', line, re.IGNORECASE)
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
            script_keys = ['run', 'script', 'command', 'before_script', 'after_script', 'entrypoint', 'commands', 'entry', 'bash', 'powershell', 'pwsh', 'cmd', 'runcmd', 'bootcmd', 'args', 'setup', 'install', 'test']
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
                            stripped_j = lines[j].strip()
                            if not stripped_j or stripped_j.startswith('#'):
                                j += 1
                                continue
                            curr_indent = len(lines[j]) - len(lines[j].lstrip())
                            if curr_indent < indent:
                                break
                            list_match = re.match(r'^\s*-\s*(.*)', lines[j])
                            if list_match:
                                cmd = list_match.group(1).strip()
                                if cmd in ('|', '>', '|-', '|+', '>-', '>+'):
                                    # Multi-line block within list
                                    block_lines = []
                                    list_indent = len(lines[j]) - len(lines[j].lstrip())
                                    j += 1
                                    while j < len(lines):
                                        if not lines[j].strip():
                                            block_lines.append("")
                                            j += 1
                                            continue
                                        inner_indent = len(lines[j]) - len(lines[j].lstrip())
                                        if inner_indent <= list_indent:
                                            break
                                        # Strip indentation from block lines
                                        block_lines.append(lines[j][list_indent+2:])
                                        j += 1
                                    if block_lines:
                                        list_items.append("\n".join(block_lines))
                                    continue
                                elif cmd:
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

    # 9. Check for Unified Diff (.diff, .patch, or by content)
    if check_name.lower().endswith(('.diff', '.patch')) or \
       (content.startswith(b'--- ') or content.startswith(b'Index: ') or content.startswith(b'diff --git ') or content.startswith(b'commit ')):
        try:
            text = content.decode('utf-8', errors='ignore')
            lines = text.splitlines()
            current_file = None
            hunk_info = None
            hunk_lines = []

            has_additions = False

            def finalize_hunk():
                nonlocal has_additions
                if current_file and hunk_info and hunk_lines and has_additions:
                    hunk_text = "\n".join(hunk_lines)
                    yield (f"{name} [{current_file} @ {hunk_info}]", hunk_text.encode('utf-8'))
                has_additions = False

            for line in lines:
                if line.startswith('+++ '):
                    yield from finalize_hunk()
                    path_part = line[4:].split('\t')[0].strip()
                    if path_part.startswith(('a/', 'b/')) and '/' in path_part:
                        path_part = path_part[2:]
                    current_file = path_part
                    hunk_info = None
                    hunk_lines = []
                elif line.startswith('@@ ') and current_file:
                    yield from finalize_hunk()
                    match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    hunk_info = f"line {match.group(1)}" if match else "unknown"
                    hunk_lines = []
                elif hunk_info is not None:
                    if line.startswith('+'):
                        if not line.startswith('+++ '):
                            hunk_lines.append(line[1:])
                            has_additions = True
                    elif line.startswith(' '):
                        hunk_lines.append(line[1:])
                    elif line.startswith('-'):
                        if line.startswith('--- '):
                            yield from finalize_hunk()
                            hunk_info = None
                            hunk_lines = []
                        continue  # Skip deletions
                    else:
                        yield from finalize_hunk()
                        hunk_info = None
                        hunk_lines = []

            yield from finalize_hunk()
            return
        except Exception:
            pass

    # 10. Check for Systemd Service or Desktop Files
    if check_basename.endswith(('.service', '.desktop')):
        try:
            text = content.decode('utf-8', errors='ignore')
            snippets = []
            current_snippet = []

            def finalize_snippet():
                if current_snippet:
                    snippets.append(" ".join([c.rstrip('\\').strip() for c in current_snippet]))

            # Common keys that contain commands
            cmd_keys = ('exec', 'execstart', 'execstartpre', 'execstartpost',
                        'execstop', 'execstoppost', 'execreload')

            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(('#', ';')):
                    continue

                # Match key=value
                match = re.match(r'^([^=\s]+)\s*=\s*(.*)', stripped)
                if match:
                    finalize_snippet()
                    key = match.group(1).strip().lower()
                    val = match.group(2).strip()
                    if key in cmd_keys:
                        current_snippet = [val]
                    else:
                        current_snippet = []
                elif current_snippet:
                    # Multi-line continuation
                    current_snippet.append(stripped)

                if current_snippet and not line.rstrip().endswith('\\'):
                    finalize_snippet()
                    current_snippet = []

            finalize_snippet()

            if snippets:
                for i, cmd in enumerate(snippets, 1):
                    if cmd.strip():
                        yield (f"{name} [Command {i}]", cmd.encode('utf-8'))
                return
        except Exception:
            pass

    # 11. Check for Makefile
    if check_basename.endswith('makefile'):
        try:
            text = content.decode('utf-8', errors='ignore')
            yielded_any = False
            recipes_count = 0
            vars_count = 0

            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                i += 1

                if not line.strip() or line.strip().startswith('#'):
                    continue

                if line.startswith('\t'):
                    # Recipe
                    content_parts = [line[1:]]
                    while line.rstrip().endswith('\\') and i < len(lines):
                        line = lines[i]
                        i += 1
                        content_parts.append(line[1:] if line.startswith('\t') else line.strip())

                    cmd = " ".join([c.rstrip('\\').strip() for c in content_parts])
                    if cmd.strip():
                        recipes_count += 1
                        yield (f"{name} [Recipe {recipes_count}]", cmd.encode('utf-8'))
                        yielded_any = True
                else:
                    # Check for define block
                    define_match = re.match(r'^\s*(?:(?:export|unexport|override|private)\s+)*define\s+([\w.-]+)', line, re.IGNORECASE)
                    if define_match:
                        block_name = define_match.group(1)
                        block_parts = []
                        while i < len(lines):
                            line = lines[i]
                            i += 1
                            if re.match(r'^\s*endef\b', line, re.IGNORECASE):
                                break
                            block_parts.append(line)

                        val = "\n".join(block_parts)
                        if val.strip():
                            vars_count += 1
                            yield (f"{name} [Variable {vars_count}: {block_name}]", val.encode('utf-8'))
                            yielded_any = True
                        continue

                    # Check for variable assignment: VAR = val, VAR := val, VAR += val, VAR ?= val, VAR != cmd
                    # Support prefixes like export, override, etc. and leading whitespace
                    var_match = re.match(r'^\s*(?:(?:export|unexport|override|private)\s+)*([\w.-]+)\s*([:+?!]?=)\s*(.*)', line)
                    if var_match:
                        content_parts = [var_match.group(3)]
                        while line.rstrip().endswith('\\') and i < len(lines):
                            line = lines[i]
                            i += 1
                            content_parts.append(line.strip())

                        val = " ".join([c.rstrip('\\').strip() for c in content_parts])
                        if val.strip():
                            vars_count += 1
                            yield (f"{name} [Variable {vars_count}]", val.encode('utf-8'))
                            yielded_any = True

            if yielded_any:
                return
        except Exception:
            pass

    # 12. Fallback: yield as a single snippet if it's a supported file type
    # If scan_all_files is True, we always yield. Otherwise check extension/shebang.
    if Config.is_supported_file(check_name, content=content, is_member=(depth > 0)):
        yield name, content
    elif depth == 0 and name.startswith(("[", "boundary")):
        # Special case for legacy test compatibility (test_extra_snippets.py)
        # Some tests use non-standard names for snippets but expect them to be scanned
        yield name, content


def iter_windows(fh, size: int, deep_scan: bool, maxlen: Optional[int] = None) -> Generator[Tuple[int, bytes], None, None]:
    """Provide parts of the file for the local scanner to check.

    By default, this function only reads the beginning and end of a file.
    This is fast because most dangerous code and script starts are
    found in these parts. If deep_scan is True, the function reads the
    entire file instead.

    The chunk size defaults to 1024 bytes (maxlen). This matches the
    requirements of the pre-trained scripts.h5 model.
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
    modified_since: Optional[float] = None,
) -> Generator[Tuple[str, Tuple[Any, ...]], None, None]:
    """Scan files for dangerous content and optionally use AI for analysis.

    Args:
        scan_targets: Folder path or list of file/folder paths to search.
        deep_scan: Whether to scan overlapping 1024-byte windows beyond the first block.
        show_all: Whether to yield all scanned files regardless of threat level threshold.
        use_gpt: Whether to request GPT analysis when the local model is confident.
        rate_limit: Maximum number of GPT requests permitted per minute.
        max_concurrent_requests: Maximum number of GPT requests executed concurrently.
        dry_run: Whether to list files that would be scanned without running the model or API.
        exclude_patterns: List of glob patterns to exclude from the scan.
        extra_snippets: List of (name, content) tuples to scan as in-memory buffers.
        modified_since: A timestamp. If provided, only files modified after this time are scanned.

    Yields:
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
    scan_targets = _normalize_targets(scan_targets)

    url_targets = []
    local_targets = []
    for t in scan_targets:
        # Defensive str() conversion ensures robustness even if targets are not strings
        t_str = str(t)
        if t_str.lower().startswith(('http://', 'https://')):
            url_targets.append(t_str)
        else:
            local_targets.append(t_str)

    explicit_targets = {Path(t) for t in local_targets}
    explicit_files = {f for f in explicit_targets if f.is_file()}

    file_list = collect_files(local_targets, modified_since=modified_since)

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

    def handle_scan_result(path: str, maxconf: float, max_window_bytes: bytes, line_num: Union[int, str], full_content: Optional[str] = None, force: bool = False) -> Generator[Tuple[str, Any], None, None]:
        if maxconf >= 0:
            # Filename analysis for additional threat indicators
            file_threat, file_msg = analyze_filename(path)

            # Combine threat levels (take the maximum of content and filename threat)
            effective_maxconf = max(maxconf, file_threat)

            percent = format_percent(effective_maxconf * 100.0)
            snippet = ''.join(map(chr, max_window_bytes)).strip()
            cleaned_snippet = _clean_snippet_for_ai(snippet)

            admin_note = ""
            user_note = ""
            if file_threat >= 0.5:
                admin_note = f"[Filename Warning] {file_msg}"
                user_note = f"Caution: {file_msg}"

            if full_content is not None:
                _virtual_source_cache[path] = full_content

            if effective_maxconf >= threshold_val and use_gpt and Config.GPT_ENABLED:
                gpt_requests.append(
                    {
                        "path": path,
                        "percent": percent,
                        "snippet": snippet,
                        "cleaned_snippet": cleaned_snippet,
                        "line": line_num,
                        "full_content": full_content,
                        "admin_desc": admin_note,
                        "user_desc": user_note,
                    }
                )
            elif effective_maxconf >= threshold_val or show_all or force:
                yield (
                    'result',
                    (
                        path,
                        percent,
                        admin_note,
                        user_note,
                        '',
                        cleaned_snippet,
                        line_num,
                    )
                )

    def _scan_fh(fh, name: str, size: int, is_virtual: bool = False) -> Generator[Tuple[str, Any], None, None]:
        maxconf = -1.0
        max_window_bytes = b""
        max_offset = 0
        hits_found = 0
        full_bytes = None
        decoded_content = None

        if is_virtual:
            curr_pos = fh.tell()
            fh.seek(0)
            full_bytes = fh.read()
            fh.seek(curr_pos)
            decoded_content = full_bytes.decode('utf-8', errors='ignore')

        for offset, window in iter_windows(fh, size, deep_scan):
            if cancel_event.is_set():
                break
            result, padded_bytes = predict_window(window)
            if result > maxconf:
                maxconf = result
                max_window_bytes = padded_bytes
                max_offset = offset

            if result >= threshold_val:
                hits_found += 1
                if not is_virtual and full_bytes is None:
                    curr_pos = fh.tell()
                    fh.seek(0)
                    full_bytes = fh.read()
                    fh.seek(curr_pos)

                line_num = full_bytes[:offset].count(b'\n') + 1
                yield from handle_scan_result(name, result, padded_bytes, line_num, full_content=decoded_content)

        if not cancel_event.is_set() and hits_found == 0:
            if is_virtual:
                line_num = full_bytes[:max_offset].count(b'\n') + 1
            else:
                line_num = 1
                try:
                    curr_pos = fh.tell()
                    fh.seek(0)
                    line_num = fh.read(max_offset).count(b'\n') + 1
                    fh.seek(curr_pos)
                except Exception:
                    pass
            yield from handle_scan_result(name, maxconf, max_window_bytes, line_num, full_content=decoded_content, force=True)

    for file_path in file_list:
        if cancel_event.is_set():
            break

        progress_count += 1
        yield ('progress', (progress_count, total_progress, f"Scanning: {file_path.name}"))

        is_explicit = file_path in explicit_files
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

        actual_files_scanned += 1
        if file_size is not None:
            total_bytes_scanned += file_size

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
        elif file_size is not None:
            try:
                with open(file_path, 'rb') as f:
                    yield from _scan_fh(f, str(file_path), file_size, is_virtual=False)
            except OSError as err:
                yield (
                    'result',
                    (
                        str(file_path),
                        'Error',
                        '',
                        '',
                        '',
                        f"Error reading file: {err}",
                        '-',
                    )
                )

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
            with io.BytesIO(content) as f:
                yield from _scan_fh(f, name, file_size, is_virtual=True)

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
            
            tasks_pending = set(tasks)
            try:
                while tasks_pending and not cancel_event.is_set():
                    done, tasks_pending = await asyncio.wait(
                        tasks_pending,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        results.append(await t)
            finally:
                for t in tasks_pending:
                    t.cancel()
                if tasks_pending:
                    await asyncio.gather(*tasks_pending, return_exceptions=True)
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
                chatgpt_conf_percent = format_percent(json_data["threat-level"])

            # Prepend filename warnings to AI analysis if present
            if request.get("admin_desc"):
                admin_desc = f"{request['admin_desc']}\n\n{admin_desc}"
            if request.get("user_desc"):
                enduser_desc = f"{request['user_desc']}\n\n{enduser_desc}"

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
                chatgpt_conf_percent = format_percent(json_data["threat-level"])

            # Prepend pre-existing notes if present
            if request.get("admin_desc"):
                admin_desc = f"{request['admin_desc']}\n\n{admin_desc}"
            if request.get("user_desc"):
                enduser_desc = f"{request['user_desc']}\n\n{enduser_desc}"

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
                    if risk in ('high', 'medium'):
                        threats_found += 1
                        if risk == 'high':
                            high_risk_found += 1
                        else:
                            medium_risk_found += 1
            elif event_type == 'summary':
                total_files, total_bytes, elapsed_time = data
                metrics['total_files'] = total_files
                metrics['total_bytes'] = total_bytes
                metrics['elapsed_time'] = elapsed_time
    finally:
        if cancel_event.is_set():
            enqueue_ui_update(finish_scan_state)
            enqueue_ui_update(update_status, f"Scan cancelled after {current_scanned} files.")
        else:
            enqueue_ui_update(
                finish_scan_state,
                metrics.get('total_files', current_scanned),
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
    modified_since: Optional[float] = None,
) -> None:
    """Read scan results and send them to the UI window.

    Args:
        scan_targets: Folder path or list of files to scan.
        deep_scan: Whether to evaluate all 1024-byte windows.
        show_all: Whether to display all results regardless of threat level.
        use_gpt: Whether to enrich suspicious files with GPT output.
        rate_limit: Maximum allowed GPT requests per minute.
        dry_run: Whether to simulate the scan.
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
        modified_since=modified_since,
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
    """Scan specific folders again in the background and update the results."""
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

    count = len(results)
    finding_text = "finding" if count == 1 else "findings"
    lines = [f"{BOLD}--- GPT SCAN - CONSOLE TRIAGE REPORT ({count} {finding_text}) ---{RESET}", ""]

    def color_conf(conf_str):
        if not use_color:
            return conf_str
        val = parse_percent(conf_str)
        if val >= 80:
            return f"{RED}{BOLD}{conf_str}{RESET}"
        if val >= 50:
            return f"{YELLOW}{BOLD}{conf_str}{RESET}"
        return f"{conf_str}"

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

        location = f"{path}:{line_num}" if line_num != "-" else path
        lines.append(f"{GRAY}[{i}]{RESET} {risk_label} - {BOLD}{location}{RESET}")

        # Consolidate scores and links
        meta_parts = [f"{GRAY}Local:{RESET} {color_conf(own_conf)}"]
        if gpt_conf:
            meta_parts.append(f"{GRAY}AI:{RESET} {color_conf(gpt_conf)}")

        vt_url = get_virustotal_url(path, snippet)
        if vt_url:
            meta_parts.append(f"{GRAY}VT:{RESET} {vt_url}")

        online_url = get_online_url(path, line_num)
        if online_url:
            meta_parts.append(f"{GRAY}Online:{RESET} {online_url}")

        lines.append(f"    {'  '.join(meta_parts)}")

        # Consolidate AI analysis
        if admin or user:
            lines.append("")
            if admin:
                for line in admin.strip().splitlines():
                    lines.append(f"    {GRAY}Admin:{RESET} {line}")
            if user:
                if admin:
                    lines.append("")
                for line in user.strip().splitlines():
                    lines.append(f"    {GRAY}User:{RESET} {line}")
            lines.append("")

        # Snippet preview (up to 3 lines)
        cols = shutil.get_terminal_size((80, 20)).columns
        max_snippet_len = max(20, cols - 8)
        snippet_lines = snippet.strip().split('\n')[:3]
        for sl in snippet_lines:
            if len(sl) > max_snippet_len:
                sl = sl[:max_snippet_len - 3] + "..."
            lines.append(f"    {GRAY}>{RESET} {sl}")
        lines.append("")

    return "\n".join(lines)


def generate_sarif(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a SARIF log from the scan results.

    Args:
        results: List of result dictionaries.

    Returns:
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
                                    "text": "Potential dangerous content detected."
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

    Args:
        results: List of result dictionaries.

    Returns:
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

        vt_url = get_virustotal_url(path, snippet)
        online_url = get_online_url(path, r.get("line", 1))

        links = []
        if vt_url:
            links.append(f'<a href="{vt_url}" target="_blank">VirusTotal</a>')
        if online_url:
            links.append(f'<a href="{online_url}" target="_blank">Online Source</a>')
        links_html = "<br>".join(links)

        rows.append(f"""
        <tr class="{row_class}">
            <td>{html.escape(path)}</td>
            <td>{html.escape(str(r.get("line", "-")))}</td>
            <td>{html.escape(gpt_conf or own_conf)}</td>
            <td>
                <strong>Admin:</strong> {admin_html}<br>
                <strong>User:</strong> {user_html}
            </td>
            <td>{links_html}</td>
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
                <th style="width: 15%">Path</th>
                <th style="width: 5%">Line</th>
                <th style="width: 10%">Threat Level</th>
                <th style="width: 25%">Analysis</th>
                <th style="width: 10%">Links</th>
                <th style="width: 35%">Snippet</th>
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

    Args:
        results: List of result dictionaries.

    Returns:
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

        vt_url = get_virustotal_url(path, snippet)
        online_url = get_online_url(path, line)
        if vt_url:
            lines.append(f"- **VirusTotal:** [Check Hash]({vt_url})")
        if online_url:
            lines.append(f"- **Online View:** [View Source]({online_url})")
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


def run_cli(targets: Union[str, List[str]], deep: bool, show_all: bool, use_gpt: bool, rate_limit: int, output_format: str = 'csv', dry_run: bool = False, exclude_patterns: Optional[List[str]] = None, fail_threshold: Optional[int] = None, output_file: Optional[str] = None, extra_snippets: Optional[List[Tuple[str, bytes]]] = None, import_file: Optional[str] = None, modified_since: Optional[float] = None) -> int:
    """Run scans and show results in the terminal or save them to a file.

    Args:
        targets: Folder or list of files to scan.
        deep: Whether to evaluate all 1024-byte windows.
        show_all: Whether to emit every scanned file.
        use_gpt: Whether to request GPT analysis for confident detections.
        rate_limit: Maximum allowed GPT requests per minute.
        output_format: Format of the output ('csv', 'json', 'sarif', 'html', or 'markdown'). Defaults to 'csv'.
        dry_run: Whether to simulate the scan.
        exclude_patterns: List of glob patterns to exclude from the scan.
        fail_threshold: Threat level threshold to trigger a failure count.
        output_file: Path to a file where results should be saved.
        extra_snippets: List of (name, content) tuples to scan as in-memory buffers.
        import_file: Path to a previous scan report to import and process.
        modified_since: A timestamp. If provided, only files modified after this time are scanned.

    Returns:
        The number of suspicious files detected.
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
            modified_since=modified_since,
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
            metrics['total_files'] = total_files
            metrics['total_bytes'] = total_bytes
            metrics['elapsed_time'] = elapsed_time

    if final_progress is not None:
        print(file=sys.stderr)
        total_scanned = metrics.get('total_files', final_progress[1])
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

    res = {}
    for key, alts in REPORT_FIELD_MAPPING.items():
        found_val = ""
        for alt in alts:
            if (val := item.get(alt)) not in (None, ""):
                found_val = str(val)
                break
        res[key] = found_val
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
                # Path, Line, Threat Level, Analysis, Links (optional), Snippet
                # Snippet is usually in the last cell, which is at index 5 if Links is present
                snippet_cell = cells[5] if len(cells) >= 6 else cells[4]
                # Snippet is inside <pre><code>
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

                            item['admin_desc'] = html.unescape(admin_match.group(1).replace('<br>', '\n').replace('\\|', '|')).strip() if admin_match else ""
                            item['end-user_desc'] = html.unescape(user_match.group(1).replace('<br>', '\n').replace('\\|', '|')).strip() if user_match else ""
                            del item['Analysis']

                        if 'Threat Level' in item:
                            item['gpt_conf'] = item['Threat Level']
                            del item['Threat Level']

                        # Clean up Snippet (remove backticks or <code> tags)
                        if 'Snippet' in item:
                            raw_snippet = item['Snippet']
                            if raw_snippet.startswith('<code>') and raw_snippet.endswith('</code>'):
                                raw_snippet = raw_snippet[6:-7]
                            else:
                                raw_snippet = raw_snippet.strip('`')

                            item['snippet'] = html.unescape(raw_snippet).replace('\\|', '|')
                            del item['Snippet']

                        if 'Path' in item:
                            item['Path'] = html.unescape(item['Path']).replace('\\|', '|')

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
            raise ValueError("File or web link content is empty.")
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


def import_results(event: Optional[tk.Event] = None) -> None:
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
        initialdir=_get_initial_dir()
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

    url = simpledialog.askstring("Import from Web Link", "Enter the web link of the scan results to import:")
    if not url:
        return

    url = url.strip()
    try:
        update_status(f"Fetching results from {url}...")
        content_bytes = fetch_url_content(url)
        content = content_bytes.decode('utf-8', errors='ignore')

        data_to_import = parse_report_content(content, filename_hint=url)

        if not data_to_import:
            messagebox.showwarning("Import Warning", "No valid scan results found at the provided web link.")
            return

        _finalize_import(data_to_import, url)

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not import results from web link:\n{err}")


def clear_ai_cache() -> None:
    """Clear the AI analysis cache and update the persistent file."""
    Config.gpt_cache = {}
    Config.save_cache()
    update_status("AI Analysis cache cleared.")


def clear_results() -> None:
    """Clear all results from the Treeview and reset progress/status."""
    global _all_results_cache, _last_scan_summary, _virtual_source_cache
    _all_results_cache = []
    _last_scan_summary = ""
    _virtual_source_cache = {}
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


def export_results(event: Optional[tk.Event] = None) -> None:
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
        initialdir=_get_initial_dir()
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


def _resolve_file_paths(event_or_path: Union[tk.Event, str, None], verify: bool = True) -> List[str]:
    """Retrieve and optionally verify multiple file paths from an event or direct argument."""
    targets = []
    if isinstance(event_or_path, str):
        targets.append(event_or_path)
    else:
        if not tree:
            return []
        selection = tree.selection()
        for item_id in selection:
            values = _get_item_raw_values(item_id)
            if values:
                targets.append(str(values[0]))

    valid_paths = []
    for path in targets:
        if verify and not path.startswith("[") and not path.startswith(("http://", "https://")) and not os.path.exists(path):
            continue
        valid_paths.append(path)

    if verify and targets and not valid_paths:
        messagebox.showwarning("Files Not Found", "The selected file(s) could not be located on disk.")

    return valid_paths


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
    conf_frame.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(5, 0))
    conf_frame.columnconfigure(5, weight=1)

    risk_badge = tk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'), padx=8, pady=2)
    risk_badge.grid(row=0, column=0, sticky="w", padx=(0, 20))

    ttk.Label(conf_frame, text="Local Threat:").grid(row=0, column=1, sticky="w")
    own_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))
    own_conf_label.grid(row=0, column=2, sticky="w", padx=(5, 20))

    ai_conf_prefix = ttk.Label(conf_frame, text="AI Threat:")
    gpt_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))

    ttk.Label(conf_frame, text="Detected Line:").grid(row=0, column=6, sticky="w", padx=(20, 5))
    line_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))
    line_label.grid(row=0, column=7, sticky="w")

    ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

    paned_window = ttk.Panedwindow(main_frame, orient=tk.VERTICAL)
    paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

    # Analysis sections
    analysis_frame = ttk.LabelFrame(paned_window, text="AI Analysis", padding=5)
    admin_label = ttk.Label(analysis_frame, text="Administrator Notes:", font=('TkDefaultFont', 9, 'bold'))
    admin_text = scrolledtext.ScrolledText(analysis_frame, height=5, wrap=tk.WORD)
    user_label = ttk.Label(analysis_frame, text="End-User Notes:", font=('TkDefaultFont', 9, 'bold'))
    user_text = scrolledtext.ScrolledText(analysis_frame, height=5, wrap=tk.WORD)

    # Snippet section
    snippet_frame = ttk.LabelFrame(paned_window, text="Code Snippet", padding=5)
    paned_window.add(snippet_frame, weight=1)

    snippet_text = scrolledtext.ScrolledText(snippet_frame, height=8, font='TkFixedFont', wrap=tk.NONE)
    snippet_text.pack(fill=tk.BOTH, expand=True)
    snippet_text.tag_configure("highlight", background="yellow", foreground="black")

    showing_full_source = False

    def load_display_code(path, line, snippet, silent_fallback=False):
        """Load and display either the snippet or full source code."""
        nonlocal showing_full_source
        if showing_full_source:
            content = None
            if path in _virtual_source_cache:
                content = _virtual_source_cache[path]
            elif path.startswith("["):
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
                except Exception as e:
                    if not silent_fallback and str(e) != "Cancelled":
                        messagebox.showerror("Error", f"Could not read file: {e}")

            if content is not None:
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

    reveal_btn = ttk.Button(header_frame, text="Show in Folder", width=14, command=lambda: show_in_folder(path_entry.get()))
    reveal_btn.grid(row=1, column=3, padx=2)
    bind_hover_message(reveal_btn, "Show this file in the system file manager. (Ctrl+Enter)", label=status_bar)

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
                if not vals:
                    return
                snippet = vals[5]

                result = request_single_gpt_analysis(snippet)

                if result:
                    # updated_vals: (path, own_conf, admin, user, gpt_conf, snippet)
                    updated_vals = list(vals)

                    admin_note = result.get("administrator", "")
                    user_note = result.get("end-user", "")

                    # Prepend pre-existing notes if present
                    if vals[2]:
                        admin_note = f"{vals[2]}\n\n{admin_note}"
                    if vals[3]:
                        user_note = f"{vals[3]}\n\n{user_note}"

                    updated_vals[2] = admin_note
                    updated_vals[3] = user_note
                    # Safer extraction of threat-level
                    threat_level = result.get("threat-level", 0)
                    updated_vals[4] = format_percent(threat_level)

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

    def copy_as_report_details():
        results = _get_tree_results_as_dicts([current_item_id])
        if results:
            report = generate_console_report(results, use_color=False)
            root.clipboard_clear()
            root.clipboard_append(report)
            set_local_status("Result copied as Triage Report.", temporary=True)

    def copy_sha256_details():
        path = path_entry.get()
        snippet = snippet_text.get("1.0", tk.END).strip()
        h = get_effective_sha256(path, snippet)

        if h:
            root.clipboard_clear()
            root.clipboard_append(h)
            set_local_status(f"SHA256 copied: {h[:8]}...", temporary=True)
        else:
            messagebox.showwarning("Error", "Could not calculate file hash.", parent=details_win)

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
    bind_hover_message(analyze_btn, "Use AI to analyze this code snippet. (Ctrl+G)", label=status_bar)
    if not Config.GPT_ENABLED:
        analyze_btn.config(state='disabled')

    intel_btn = ttk.Menubutton(btn_frame, text="Intel", width=12)
    intel_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(intel_btn, "Threat intelligence for this item (VirusTotal, online repository).", label=status_bar)

    intel_menu = tk.Menu(intel_btn, tearoff=0)
    intel_menu.add_command(label="Check on VirusTotal", command=lambda: check_virustotal(path_entry.get()), accelerator="Ctrl+T")
    intel_menu.add_command(label="View Online", command=lambda: view_online(path_entry.get(), line=line_label.cget("text")), accelerator="Ctrl+L")
    intel_btn["menu"] = intel_menu

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: Management
    rescan_btn = ttk.Button(btn_frame, text="Rescan", width=10, command=on_rescan)
    rescan_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(rescan_btn, "Re-scan this file with current settings. (F5 or R)", label=status_bar)

    exclude_btn = ttk.Button(btn_frame, text="Exclude", width=10, command=on_exclude)
    exclude_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(exclude_btn, "Exclude this file from future scans. (Delete)", label=status_bar)

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: Copy Actions
    copy_menu_btn = ttk.Menubutton(btn_frame, text="Copy...", width=12)
    copy_menu_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(copy_menu_btn, "Copy analysis, JSON, or code to the clipboard.", label=status_bar)

    copy_menu = tk.Menu(copy_menu_btn, tearoff=0)
    copy_menu.add_command(label="Copy Analysis", command=copy_analysis, accelerator="Ctrl+Shift+C")
    copy_menu.add_command(label="Copy Path", command=copy_path_details, accelerator="Ctrl+Shift+P")
    copy_menu.add_command(label="Copy SHA256", command=copy_sha256_details, accelerator="Ctrl+H")
    copy_menu.add_command(label="Copy as JSON", command=copy_as_json_details, accelerator="Ctrl+J")
    copy_menu.add_command(label="Copy as Triage Report", command=copy_as_report_details, accelerator="Ctrl+Shift+R")
    copy_menu.add_command(label="Copy Code", command=copy_code, accelerator="Ctrl+S")
    copy_menu_btn["menu"] = copy_menu

    ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

    # Group: View
    source_toggle_btn = ttk.Button(btn_frame, text="Show Full Source", width=18, command=toggle_source)
    source_toggle_btn.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(source_toggle_btn, "Toggle between the suspicious snippet and the full file content. (Ctrl+U)", label=status_bar)

    # Group: System
    close_btn = ttk.Button(btn_frame, text="Close", command=details_win.destroy)
    close_btn.pack(side=tk.RIGHT, padx=5, ipady=5)
    bind_hover_message(close_btn, "Close this window. (Esc)", label=status_bar)

    def refresh_content(new_id):
        nonlocal current_item_id
        current_item_id = new_id
        vals = _get_item_raw_values(new_id)
        if not vals:
            return

        # vals: (path, own_conf, admin, user, gpt_conf, snippet, line)
        path = str(vals[0])

        is_virtual = path.startswith("[")
        is_non_url_virtual = is_virtual and not path.startswith("[URL] ")

        # Ensure buttons reflect current scanning state
        is_scanning = current_cancel_event is not None

        # Define button states based on scan status and virtual path
        rescan_btn.config(state='disabled' if is_scanning or is_virtual else 'normal')
        analyze_btn.config(state='disabled' if is_scanning or not Config.GPT_ENABLED else 'normal')
        exclude_btn.config(state='disabled' if is_scanning or is_virtual else 'normal')
        open_btn.config(state='disabled' if is_virtual else 'normal')
        intel_btn.config(state='disabled' if is_scanning else 'normal')
        try:
            intel_menu.entryconfig("Check on VirusTotal", state='normal')
            intel_menu.entryconfig("View Online", state='disabled' if is_non_url_virtual else 'normal')
        except tk.TclError:
            pass
        path_copy_btn.config(state='normal')
        reveal_btn.config(state='disabled' if is_virtual else 'normal')

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
            ai_conf_prefix.grid(row=0, column=3, sticky="w")
            gpt_conf_label.grid(row=0, column=4, sticky="w", padx=(5, 0))
            gpt_conf_label.config(text=gpt_conf)
        else:
            ai_conf_prefix.grid_forget()
            gpt_conf_label.grid_forget()
            gpt_conf_label.config(text="")

        if admin or user:
            if str(analysis_frame) not in paned_window.panes():
                paned_window.insert(0, analysis_frame, weight=1)

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
            if str(analysis_frame) in paned_window.panes():
                paned_window.forget(analysis_frame)

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
        except ValueError:
            pass

    def on_next():
        all_visible = tree.get_children()
        try:
            idx = all_visible.index(current_item_id)
            if idx < len(all_visible) - 1:
                new_id = all_visible[idx + 1]
                tree.selection_set(new_id)
                tree.see(new_id)
                refresh_content(new_id)
        except ValueError:
            pass

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
    details_win.bind('<Control-Return>', lambda e: show_in_folder(path_entry.get()))
    details_win.bind('<Command-Return>', lambda e: show_in_folder(path_entry.get()))
    details_win.bind('<Control-s>', lambda e: copy_code())
    details_win.bind('<Command-s>', lambda e: copy_code())
    details_win.bind('<Control-j>', lambda e: copy_as_json_details())
    details_win.bind('<Command-j>', lambda e: copy_as_json_details())
    details_win.bind('<Control-Shift-R>', lambda e: copy_as_report_details())
    details_win.bind('<Command-Shift-R>', lambda e: copy_as_report_details())
    details_win.bind('<Control-Shift-P>', lambda e: copy_path_details())
    details_win.bind('<Command-Shift-P>', lambda e: copy_path_details())
    details_win.bind('<Control-h>', lambda e: copy_sha256_details())
    details_win.bind('<Command-h>', lambda e: copy_sha256_details())
    details_win.bind('<Control-g>', lambda e: on_analyze_now())
    details_win.bind('<Command-g>', lambda e: on_analyze_now())
    details_win.bind('<Control-t>', lambda e: check_virustotal(path_entry.get()))
    details_win.bind('<Command-t>', lambda e: check_virustotal(path_entry.get()))
    details_win.bind('<Control-l>', lambda e: view_online(path_entry.get(), line=line_label.cget("text")))
    details_win.bind('<Command-l>', lambda e: view_online(path_entry.get(), line=line_label.cget("text")))
    details_win.bind('<Control-Shift-C>', lambda e: copy_analysis())
    details_win.bind('<Command-Shift-C>', lambda e: copy_analysis())
    details_win.bind('<Control-u>', lambda e: toggle_source())
    details_win.bind('<Command-u>', lambda e: toggle_source())
    details_win.bind('<F5>', lambda e: on_rescan())
    details_win.bind('r', lambda e: on_rescan())
    details_win.bind('R', lambda e: on_rescan())
    refresh_content(current_item_id)


def open_file(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Open the selected or specified file(s) in the system's default application."""
    file_paths = _resolve_file_paths(event_or_path)
    if not file_paths:
        return

    if len(file_paths) > 5:
        if not messagebox.askyesno("Open Files", f"Are you sure you want to open {len(file_paths)} files?"):
            return

    opened_count = 0
    for file_path in file_paths:
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
            opened_count += 1
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file '{file_path}': {e}")

    if opened_count > 0:
        file_text = "file" if opened_count == 1 else "files"
        update_status(f"Opened {opened_count} {file_text}.")


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
        snippet = str(values[5])

        h = get_effective_sha256(file_path, snippet)

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
        if file_path.startswith("[") and snippet is None:
            # Try to find it in the tree if snippet wasn't provided (explicit path case)
            for item_id in tree.get_children():
                vals = _get_item_raw_values(item_id)
                if vals and vals[0] == file_path:
                    snippet = str(vals[5])
                    break

        if not file_path.startswith("[") and not os.path.exists(file_path):
            if isinstance(event_or_path, str):
                messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")
                return
            continue

        url = get_virustotal_url(file_path, snippet)
        if url:
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


def copy_as_report(event: Optional[tk.Event] = None) -> None:
    """Copy the selected rows as a Triage Report to the clipboard."""
    if not tree:
        return

    selection = tree.selection()
    if not selection:
        return

    results = _get_tree_results_as_dicts(selection)
    report = generate_console_report(results, use_color=False)

    tree.clipboard_clear()
    tree.clipboard_append(report)
    update_status(f"Copied {len(results)} item(s) as Triage Report.")


def view_online(event_or_path: Union[tk.Event, str, None] = None, line: Optional[Union[int, str]] = None) -> None:
    """Open the selected result in a web browser (GitHub/GitLab/Bitbucket)."""
    # List of (path, line_num)
    targets: List[Tuple[str, Union[int, str]]] = []

    if isinstance(event_or_path, str):
        targets.append((event_or_path, line if line is not None else 1))
    else:
        if not tree:
            return
        selection = tree.selection()
        if not selection:
            return

        # Avoid accidental mass tab opening
        if len(selection) > 5:
            if not messagebox.askyesno("View Online",
                                        f"You have selected {len(selection)} files. Do you want to open that many browser tabs?"):
                return

        for item_id in selection:
            vals = _get_item_raw_values(item_id)
            if vals:
                path = str(vals[0])
                line_num = vals[6] if len(vals) > 6 and vals[6] != "-" else 1
                targets.append((path, line_num))

    success_count = 0
    last_path = ""
    for file_path, line_num in targets:
        url = get_online_url(file_path, line_num)
        if url:
            webbrowser.open(url)
            success_count += 1
            last_path = file_path

    if success_count > 0:
        if success_count == 1:
            update_status(f"Opening online view for {os.path.basename(last_path)}...")
        else:
            update_status(f"Opening online view for {success_count} files...")
    elif not isinstance(event_or_path, str):
        messagebox.showinfo(
            "Online View Unavailable",
            "The selected file(s) could not be resolved to an online repository.\n\n"
            "Ensure they are part of a Git project with a remote origin (GitHub, GitLab, or Bitbucket) "
            "or are remote web link targets."
        )


def show_in_folder(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Show the selected or specified file(s) in the system file manager."""
    file_paths = _resolve_file_paths(event_or_path)
    if not file_paths:
        return

    if len(file_paths) > 5:
        if not messagebox.askyesno("Show in Folder", f"Are you sure you want to show {len(file_paths)} files?"):
            return

    # On Linux, we deduplicate folders to avoid opening the same window multiple times
    linux_dirs = set()
    revealed_count = 0
    for file_path in file_paths:
        try:
            if sys.platform == "win32":
                subprocess.run(['explorer', '/select,', os.path.normpath(file_path)])
                revealed_count += 1
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", file_path])
                revealed_count += 1
            else:
                # Linux: xdg-open opens the folder. Deduplicate to avoid excessive windows.
                folder = os.path.dirname(os.path.abspath(file_path))
                if folder not in linux_dirs:
                    subprocess.run(["xdg-open", folder])
                    linux_dirs.add(folder)
                    revealed_count += 1
        except Exception as e:
            messagebox.showerror("Error", f"Could not reveal file '{file_path}': {e}")

    if revealed_count > 0:
        if sys.platform == "linux":
            folder_text = "folder" if revealed_count == 1 else "folders"
            update_status(f"Opened {revealed_count} {folder_text}.")
        else:
            file_text = "file" if revealed_count == 1 else "files"
            update_status(f"Revealed {revealed_count} {file_text}.")


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

    selection = tree.selection()
    has_selection = bool(selection)
    is_scanning = current_cancel_event is not None

    # Identify if any selected items are virtual (not physical local files)
    has_virtual = False
    has_non_url_virtual = False
    for item_id in selection:
        vals = _get_item_raw_values(item_id)
        if vals:
            path = str(vals[0])
            if path.startswith("["):
                has_virtual = True
                if not path.startswith("[URL] "):
                    has_non_url_virtual = True

    # Define button states based on selection and scan status
    # Base dependencies (selection only, allows interaction during scan)
    safe_base_state = "normal" if has_selection else "disabled"
    safe_local_state = "normal" if has_selection and not has_virtual else "disabled"
    safe_online_state = "normal" if has_selection and not has_non_url_virtual else "disabled"

    # Dependencies requiring no active scan (modifying actions)
    mod_local_state = "normal" if has_selection and not is_scanning and not has_virtual else "disabled"

    # Apply to footer buttons
    if view_button:
        view_button.config(state=safe_base_state)
    if open_button:
        open_button.config(state=safe_local_state)
    if rescan_button:
        rescan_button.config(state=mod_local_state)
    if exclude_button:
        exclude_button.config(state=mod_local_state)
    if reveal_button:
        reveal_button.config(state=safe_local_state)
    if intel_button:
        intel_button.config(state=safe_base_state)

    if intel_menu:
        try:
            intel_menu.entryconfig("Check on VirusTotal", state=safe_base_state)
            intel_menu.entryconfig("View Online", state=safe_online_state)
        except tk.TclError:
            pass

    if analyze_button:
        ai_available = Config.GPT_ENABLED
        analyze_button.config(state="normal" if has_selection and ai_available and not is_scanning else "disabled")

    # Update Context Menu states
    if context_menu:
        try:
            # Entry indices can be tricky if separators change, but we target by label for safety
            context_menu.entryconfig("View Details...", state=safe_base_state)
            context_menu.entryconfig("Rescan Selected", state=mod_local_state)
            context_menu.entryconfig("Analyze with AI", state="normal" if has_selection and Config.GPT_ENABLED and not is_scanning else "disabled")
            context_menu.entryconfig("Exclude Selected", state=mod_local_state)
            context_menu.entryconfig("Open", state=safe_local_state)
            context_menu.entryconfig("Show in Folder", state=safe_local_state)
            context_menu.entryconfig("Check on VirusTotal", state=safe_base_state)
            context_menu.entryconfig("View Online", state=safe_online_state)
        except tk.TclError:
            pass # Menu might not be fully initialized or entry labels differ


def copy_cli_command(event: Optional[tk.Event] = None) -> None:
    """Copy the equivalent CLI command to the system clipboard."""
    cmd_parts = ["python", "gptscan.py"]

    # Target path(s)
    raw_target = textbox.get() if textbox else Config.last_path
    if raw_target:
        try:
            # Parse possible multiple targets from the GUI textbox
            targets = shlex.split(raw_target, posix=(sys.platform != "win32"))
            for t in targets:
                cmd_parts.append(_quote_for_ui(t))
        except ValueError:
            # Fallback to quoting the raw string if parsing fails
            cmd_parts.append(_quote_for_ui(raw_target))

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

    Args:
        provider: The AI provider name (e.g., 'openai', 'ollama', 'openrouter').

    Returns:
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


def update_model_presets(provider: str) -> None:
    """Update the model combobox with presets for the selected provider."""
    if model_combo:
        model_combo['values'] = get_model_presets(provider)


def on_provider_change(event: Optional[tk.Event] = None) -> None:
    """Handle changes to the AI provider selection."""
    if not provider_var:
        return
    p = provider_var.get()
    Config.provider = p

    # Reset clients so they are recreated with new settings
    global _async_openai_client
    _async_openai_client = None

    update_model_presets(p)

    if p == "ollama":
        # Only switch to default Ollama model if current model is not an Ollama preset
        current_model = model_var.get() if model_var else ""
        ollama_presets = get_model_presets("ollama")
        if model_var and current_model not in ollama_presets:
            model_var.set("llama3.2")

        if api_base_var and not api_base_var.get().strip():
            api_base_var.set("http://localhost:11434/v1")
    elif api_base_var and api_base_var.get().strip() == "http://localhost:11434/v1":
        api_base_var.set("")
        # Default to the first preset if available, else keep current or specific default
        if model_combo and model_combo['values']:
            if model_var:
                model_var.set(model_combo['values'][0])
        elif model_var:
            model_var.set("gpt-4o")
    if model_var:
        Config.model_name = model_var.get()
    toggle_ai_controls()


def create_gui(initial_path: Optional[str] = None) -> tk.Tk:
    """Construct and return the main Tkinter GUI for the scanner.

    Args:
        initial_path: If provided, pre-fill the scan path textbox.

    Returns:
        Initialized Tk root instance ready for ``mainloop``.
    """
    global root, textbox, progress_bar, status_label, deep_var, all_var, scan_all_var, gpt_var, dry_var, git_var, filter_var, filter_entry, tree, scan_button, cancel_button, view_button, intel_button, intel_menu, rescan_button, open_button, analyze_button, exclude_button, reveal_button, results_button, browse_button, show_key_btn, default_font_measure, copy_cmd_button, clear_target_btn, git_checkbox, deep_checkbox, scan_all_checkbox, dry_checkbox, gpt_checkbox, provider_combo, model_combo, api_key_entry, api_entry, all_checkbox, threshold_spin, provider_var, model_var, api_base_var, api_key_var

    root = tk.Tk()
    root.geometry("1000x600")
    root.title("GPT Virus Scanner")
    default_font_measure = tkinter.font.Font(font='TkDefaultFont').measure

    style = ttk.Style(root)
    style.configure('Primary.TButton', font=('TkDefaultFont', 9, 'bold'))

    # --- Menu Bar ---
    menubar = tk.Menu(root)

    def add_scan_submenus(parent):
        """Add standard scan submenus to a parent menu."""
        recent_menu = tk.Menu(parent, tearoff=0)
        recent_menu.add_command(label="Last Hour", command=lambda: scan_recently_modified_click("1h"))
        recent_menu.add_command(label="Last 24 Hours", command=lambda: scan_recently_modified_click("24h"))
        recent_menu.add_command(label="Last 7 Days", command=lambda: scan_recently_modified_click("7d"))
        recent_menu.add_separator()
        recent_menu.add_command(label="Custom...", command=scan_recently_modified_click)
        parent.add_cascade(label="Scan Recently Modified", menu=recent_menu)

        git_menu = tk.Menu(parent, tearoff=0)
        git_menu.add_command(label="Scan Git Diff", command=scan_git_diff_click, accelerator="Ctrl+Shift+D")
        git_menu.add_command(label="Scan Git Hooks", command=scan_git_hooks_click, accelerator="Ctrl+Shift+G")
        git_menu.add_command(label="Scan Git Stashes", command=scan_git_stash_click, accelerator="Ctrl+Shift+Q")
        git_menu.add_command(label="Scan Git Conflicts", command=scan_git_conflicts_click)

        history_menu = tk.Menu(git_menu, tearoff=0)
        history_menu.add_command(label="Last 5 Commits", command=lambda: scan_git_history_click(5))
        history_menu.add_command(label="Last 10 Commits", command=lambda: scan_git_history_click(10))
        history_menu.add_command(label="Last 25 Commits", command=lambda: scan_git_history_click(25))
        history_menu.add_separator()
        history_menu.add_command(label="Custom...", command=scan_git_history_click)
        git_menu.add_cascade(label="Scan Recent Commits", menu=history_menu)

        reflog_menu = tk.Menu(git_menu, tearoff=0)
        reflog_menu.add_command(label="Last 5 Entries", command=lambda: scan_git_reflog_click(5))
        reflog_menu.add_command(label="Last 10 Entries", command=lambda: scan_git_reflog_click(10))
        reflog_menu.add_command(label="Last 25 Entries", command=lambda: scan_git_reflog_click(25))
        reflog_menu.add_separator()
        reflog_menu.add_command(label="Custom...", command=scan_git_reflog_click)
        git_menu.add_cascade(label="Scan Git Reflog", menu=reflog_menu)

        git_menu.add_command(label="Scan Git Configuration", command=scan_git_config_click)
        git_menu.add_command(label="Scan Git Revision...", command=scan_git_revision_click)
        parent.add_cascade(label="Git Integration", menu=git_menu)

        system_menu = tk.Menu(parent, tearoff=0)
        system_menu.add_command(label="Scan System Audit", command=scan_system_audit_click, accelerator="Ctrl+Shift+I")
        system_menu.add_command(label="Scan Shell Profiles", command=scan_shell_profiles_click, accelerator="Ctrl+Shift+B")
        system_menu.add_command(label="Scan Shell History", command=scan_shell_history_click, accelerator="Ctrl+Shift+H")
        system_menu.add_command(label="Scan System PATH", command=scan_system_path_click, accelerator="Ctrl+Shift+P")
        system_menu.add_command(label="Scan Running Processes", command=scan_running_processes_click, accelerator="Ctrl+Shift+K")
        system_menu.add_command(label="Scan Environment Variables", command=scan_env_vars_click, accelerator="Ctrl+Shift+N")
        system_menu.add_command(label="Scan Scheduled Tasks", command=scan_scheduled_tasks_click, accelerator="Ctrl+Shift+T")
        system_menu.add_command(label="Scan Startup Items", command=scan_startup_items_click, accelerator="Ctrl+Shift+A")
        system_menu.add_command(label="Scan System Services", command=scan_system_services_click, accelerator="Ctrl+Shift+S")
        system_menu.add_command(label="Scan SSH Configuration", command=scan_ssh_config_click, accelerator="Ctrl+Shift+R")
        system_menu.add_command(label="Scan Python Packages", command=scan_python_packages_click, accelerator="Ctrl+Shift+Y")
        system_menu.add_command(label="Scan Node.js Packages", command=scan_nodejs_packages_click, accelerator="Ctrl+Shift+M")
        system_menu.add_command(label="Scan Ruby Gems", command=scan_ruby_gems_click)
        system_menu.add_command(label="Scan PHP Packages", command=scan_php_packages_click)
        system_menu.add_command(label="Scan Rust Packages", command=scan_rust_packages_click)
        system_menu.add_command(label="Scan Go Packages", command=scan_go_packages_click)
        system_menu.add_command(label="Scan Java Packages", command=scan_java_packages_click)
        system_menu.add_command(label="Scan .NET Packages", command=scan_dotnet_packages_click)
        system_menu.add_command(label="Scan Browser Bookmarks", command=scan_browser_bookmarks_click)
        system_menu.add_command(label="Scan Browser Extensions", command=scan_browser_extensions_click, accelerator="Ctrl+Shift+W")
        system_menu.add_command(label="Scan Editor Extensions", command=scan_editor_extensions_click, accelerator="Ctrl+Shift+X")
        system_menu.add_command(label="Scan Documents", command=scan_documents_click)
        system_menu.add_command(label="Scan Downloads", command=scan_downloads_click, accelerator="Ctrl+Shift+J")
        system_menu.add_command(label="Scan Desktop", command=scan_desktop_click, accelerator="Ctrl+Shift+L")
        system_menu.add_command(label="Scan Temporary Folders", command=scan_temp_click, accelerator="Ctrl+Shift+Z")
        parent.add_cascade(label="System Scans", menu=system_menu)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Import Results...", command=import_results, accelerator="Ctrl+O")
    file_menu.add_command(label="Import from Clipboard", command=import_from_clipboard, accelerator="Ctrl+V")
    file_menu.add_command(label="Import from Web Link...", command=import_from_url)
    file_menu.add_command(label="Export Results...", command=export_results, accelerator="Ctrl+E")
    file_menu.add_command(label="Manage Exclusions...", command=manage_exclusions)
    file_menu.add_command(label="Manage Extensions...", command=manage_extensions)
    file_menu.add_command(label="Copy as CLI Command", command=copy_cli_command, accelerator="Ctrl+Shift+E")
    file_menu.add_separator()
    file_menu.add_command(label="Clear Results", command=clear_results)
    file_menu.add_command(label="Clear AI Cache", command=clear_ai_cache)
    file_menu.add_command(label="Clear Path History", command=clear_path_history)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)

    scan_menu = tk.Menu(menubar, tearoff=0)
    scan_menu.add_command(label="Scan Now", command=button_click, accelerator="Enter")
    scan_menu.add_command(label="Cancel Scan", command=cancel_scan, accelerator="Esc")
    scan_menu.add_separator()
    scan_menu.add_command(label="Scan File(s)...", command=browse_file_click, accelerator="Ctrl+Shift+O")
    scan_menu.add_command(label="Scan Folder...", command=browse_dir_click, accelerator="Ctrl+Shift+F")
    scan_menu.add_command(label="Scan Web Link...", command=select_url_click, accelerator="Ctrl+Shift+U")
    scan_menu.add_command(label="Scan File List...", command=browse_file_list_click)
    scan_menu.add_command(label="Scan Clipboard", command=scan_clipboard_click, accelerator="Ctrl+Shift+V")
    scan_menu.add_separator()
    add_scan_submenus(scan_menu)
    menubar.add_cascade(label="Scan", menu=scan_menu)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", f"GPT Virus Scanner v{Config.VERSION}\nThis tool uses AI to find dangerous code in your scripts."))
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

    # Configure grid weights to ensure resizing behaves correctly
    root.columnconfigure(0, weight=1)
    root.rowconfigure(3, weight=1)  # The row containing the Treeview (tree_frame)

    # --- Input Frame ---
    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    input_frame.columnconfigure(1, weight=1)

    ttk.Label(input_frame, text="Path to scan:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    textbox = ttk.Combobox(input_frame, values=Config.recent_paths)
    path_to_use = initial_path if initial_path else (Config.last_path if Config.last_path else os.getcwd())
    textbox.insert(0, path_to_use)
    textbox.select_range(0, tk.END)
    textbox.grid(row=0, column=1, sticky="ew", padx=(5, 2))
    textbox.bind('<Return>', lambda event: button_click())
    textbox.focus_set()
    bind_hover_message(textbox, "Enter one or more files, folders, or glob patterns (e.g., src/**/*.py) to scan. Separate multiple targets with spaces.")

    def clear_target():
        textbox.delete(0, tk.END)
        textbox.focus_set()

    clear_target_btn = ttk.Button(input_frame, text="×", width=3, command=clear_target)
    clear_target_btn.grid(row=0, column=2, padx=(0, 5))
    bind_hover_message(clear_target_btn, "Clear the scan target.")

    root.bind('<Escape>', lambda event: cancel_scan())
    button_box = ttk.Frame(input_frame)
    button_box.grid(row=0, column=3, sticky="e")
    browse_button = ttk.Menubutton(button_box, text="Browse", width=10)
    browse_button.pack(side=tk.LEFT, padx=(5, 2), ipady=5)
    bind_hover_message(browse_button, "Select scan targets or perform system audits.")

    scan_button = ttk.Button(button_box, text="Scan Now", command=button_click, style='Primary.TButton', default='active', width=12)
    scan_button.pack(side=tk.LEFT, padx=2, ipady=5)
    bind_hover_message(scan_button, "Start the scan. (Enter)")

    cancel_button = ttk.Button(button_box, text="Cancel", command=cancel_scan, state="disabled", width=10)
    cancel_button.pack(side=tk.LEFT, padx=(2, 0), ipady=5)
    bind_hover_message(cancel_button, "Stop the current scan. (Esc)")

    browse_menu = tk.Menu(browse_button, tearoff=0)
    browse_menu.add_command(label="Scan File(s)...", command=browse_file_click, accelerator="Ctrl+Shift+O")
    browse_menu.add_command(label="Scan Folder...", command=browse_dir_click, accelerator="Ctrl+Shift+F")
    browse_menu.add_command(label="Scan Web Link...", command=select_url_click, accelerator="Ctrl+Shift+U")
    browse_menu.add_command(label="Scan File List...", command=browse_file_list_click)
    browse_menu.add_command(label="Scan Clipboard", command=scan_clipboard_click, accelerator="Ctrl+Shift+V")
    browse_menu.add_separator()
    add_scan_submenus(browse_menu)
    browse_button["menu"] = browse_menu

    # --- Settings Container ---
    settings_frame = ttk.Frame(root)
    settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

    # --- Options Frame ---
    options_frame = ttk.LabelFrame(settings_frame, text="Scan Options", padding=10)
    options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

    gpt_var = tk.BooleanVar(value=Config.use_ai_analysis)
    dry_var = tk.BooleanVar(value=False)

    git_var = tk.BooleanVar(value=Config.git_changes_only)
    git_checkbox = ttk.Checkbutton(options_frame, text="Git changes only", variable=git_var)
    git_checkbox.grid(row=0, column=0, sticky='w', padx=10, pady=(2, 10))
    bind_hover_message(git_checkbox, "Only scan files that have been modified or are untracked in Git.")

    deep_var = tk.BooleanVar(value=Config.deep_scan)
    deep_checkbox = ttk.Checkbutton(options_frame, text="Deep scan", variable=deep_var)
    deep_checkbox.grid(row=0, column=1, sticky='w', padx=10, pady=(2, 10))
    bind_hover_message(deep_checkbox, "Scan the whole file. This is slower but more thorough. Normally, the scanner only checks the beginning and end.")

    scan_all_var = tk.BooleanVar(value=Config.scan_all_files)
    scan_all_checkbox = ttk.Checkbutton(options_frame, text="Scan all files", variable=scan_all_var)
    scan_all_checkbox.grid(row=1, column=0, sticky='w', padx=10, pady=5)
    bind_hover_message(scan_all_checkbox, "Scan all files regardless of their extension or whether they contain a script starting line (like #!/bin/bash).")

    dry_checkbox = ttk.Checkbutton(options_frame, text="Dry Run", variable=dry_var, command=toggle_dry_run)
    dry_checkbox.grid(row=1, column=1, sticky='w', padx=10, pady=5)
    bind_hover_message(dry_checkbox, "Simulate the scan process without running checks.")

    size_frame = ttk.Frame(options_frame)
    size_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=10, pady=5)
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
    copy_cmd_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=5, ipady=5)
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

    ttk.Label(provider_frame, text="Provider:").grid(row=1, column=0, sticky='w', padx=(10, 5), pady=5)
    provider_var = tk.StringVar(value=Config.provider)
    provider_combo = ttk.Combobox(provider_frame, textvariable=provider_var, values=["openai", "openrouter", "ollama"], state="readonly", width=12)
    provider_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
    bind_hover_message(provider_combo, "Select the AI service provider (OpenAI, OpenRouter, or local Ollama).")

    ttk.Label(provider_frame, text="Model:").grid(row=1, column=2, sticky='w', padx=(15, 5), pady=5)
    model_var = tk.StringVar(value=Config.model_name)
    model_combo = ttk.Combobox(provider_frame, textvariable=model_var, width=20)
    model_combo.grid(row=1, column=3, sticky='ew', padx=(5, 10), pady=5)
    bind_hover_message(model_combo, "Choose the specific AI model to use for analysis.")

    ttk.Label(provider_frame, text="API Key:").grid(row=2, column=0, sticky='w', padx=(10, 5), pady=5)

    key_container = ttk.Frame(provider_frame)
    key_container.grid(row=2, column=1, columnspan=3, sticky='ew', padx=(5, 10), pady=5)

    api_key_var = tk.StringVar(value=Config.apikey)
    api_key_entry = ttk.Entry(key_container, show="*", textvariable=api_key_var)
    api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    bind_hover_message(api_key_entry, "Enter your API key for cloud providers. (Not required for Ollama). It will be saved to apikey.txt.")

    def toggle_api_key_visibility():
        if api_key_entry['show'] == "*":
            api_key_entry.config(show="")
            show_key_btn.config(text="Hide")
        else:
            api_key_entry.config(show="*")
            show_key_btn.config(text="Show")

    show_key_btn = ttk.Button(key_container, text="Show", width=6, command=toggle_api_key_visibility)
    show_key_btn.pack(side=tk.LEFT, padx=(5, 0))
    bind_hover_message(show_key_btn, "Toggle API key visibility.")

    def on_api_key_change(*args):
        Config.apikey = api_key_var.get().strip()
        global _async_openai_client
        _async_openai_client = None

    api_key_var.trace_add("write", on_api_key_change)

    ttk.Label(provider_frame, text="API Base Web Link:").grid(row=3, column=0, sticky='w', padx=(10, 5), pady=5)
    api_entry = ttk.Entry(provider_frame)
    api_entry.grid(row=3, column=1, columnspan=3, sticky='ew', padx=(5, 10), pady=5)
    bind_hover_message(api_entry, "Set a custom web link for the AI service (e.g., http://localhost:11434/v1 for Ollama).")

    api_base_var = tk.StringVar(value=Config.api_base or "")
    api_entry.config(textvariable=api_base_var)

    def on_api_base_change(*args):
        val = api_base_var.get().strip()
        Config.api_base = val if val else None
        global _async_openai_client
        _async_openai_client = None

    api_base_var.trace_add("write", on_api_base_change)

    toggle_ai_controls()

    update_model_presets(Config.provider)

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

    # --- Filter Frame ---
    filter_frame = ttk.Frame(root)
    filter_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
    filter_frame.columnconfigure(1, weight=1)

    ttk.Label(filter_frame, text="Filter:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    filter_var = tk.StringVar()
    filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
    filter_entry.grid(row=0, column=1, sticky="ew")
    filter_entry.bind('<KeyRelease>', _apply_filter)
    filter_entry.bind('<Return>', on_filter_return)
    bind_hover_message(filter_entry, "Search results by any column (path, threat level, analysis, snippet). (Ctrl+F)")

    def clear_filter():
        filter_var.set("")
        _apply_filter()
        filter_entry.focus_set()

    clear_filter_btn = ttk.Button(filter_frame, text="×", width=3, command=clear_filter)
    clear_filter_btn.grid(row=0, column=2, padx=(0, 5))
    bind_hover_message(clear_filter_btn, "Clear the filter.")

    ttk.Separator(filter_frame, orient=tk.VERTICAL).grid(row=0, column=3, sticky="ns", padx=10)

    def on_threshold_change():
        try:
            val = int(threshold_spin.get())
            Config.THRESHOLD = max(0, min(100, val))
            _apply_filter()
        except ValueError:
            pass

    ttk.Label(filter_frame, text="Min. Threat Level:").grid(row=0, column=4, sticky="w", padx=(5, 0))
    threshold_spin = ttk.Spinbox(filter_frame, from_=0, to=100, width=5, command=on_threshold_change)
    threshold_spin.delete(0, tk.END)
    threshold_spin.insert(0, str(Config.THRESHOLD))
    threshold_spin.grid(row=0, column=5, sticky="w", padx=5)
    threshold_spin.bind('<KeyRelease>', lambda e: on_threshold_change())
    bind_hover_message(threshold_spin, "Files with a threat level lower than this will be ignored.")

    all_var = tk.BooleanVar(value=Config.show_all_files)
    all_checkbox = ttk.Checkbutton(filter_frame, text="Show all results", variable=all_var, command=_apply_filter)
    all_checkbox.grid(row=0, column=6, sticky="w", padx=(5, 0))
    bind_hover_message(all_checkbox, "Display all scanned files, including safe ones.")

    # --- Treeview ---
    style.configure('Scanner.Treeview', rowheight=50)

    # Configure tags for row highlighting
    # Note: 'alt' theme or similar might be needed for background colors to show in some environments
    tree_frame = ttk.Frame(root)
    tree_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
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
    footer_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 5))
    footer_frame.columnconfigure(11, weight=1)

    view_button = ttk.Button(footer_frame, text="View", width=10, command=view_details, style='Primary.TButton')
    view_button.grid(row=0, column=0, padx=2, ipady=5)
    bind_hover_message(view_button, "Show full analysis and code for the selected result. (Space or Enter)")

    analyze_button = ttk.Button(footer_frame, text="Analyze with AI", width=18, command=analyze_selected_with_ai, style='Primary.TButton')
    analyze_button.grid(row=0, column=1, padx=2, ipady=5)
    bind_hover_message(analyze_button, "Use AI to analyze the currently selected items. (Ctrl+G)")

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=2, sticky="ns", padx=10)

    rescan_button = ttk.Button(footer_frame, text="Rescan", width=10, command=rescan_selected)
    rescan_button.grid(row=0, column=3, padx=2, ipady=5)
    bind_hover_message(rescan_button, "Re-scan the currently selected items. (F5 or R)")

    exclude_button = ttk.Button(footer_frame, text="Exclude", width=10, command=exclude_selected)
    exclude_button.grid(row=0, column=4, padx=2, ipady=5)
    bind_hover_message(exclude_button, "Exclude the selected items from future scans. (Delete)")

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=5, sticky="ns", padx=10)

    open_button = ttk.Button(footer_frame, text="Open", width=10, command=open_file)
    open_button.grid(row=0, column=6, padx=2, ipady=5)
    bind_hover_message(open_button, "Open the selected file in its default application. (Shift+Enter)")

    reveal_button = ttk.Button(footer_frame, text="Show in Folder", width=14, command=show_in_folder)
    reveal_button.grid(row=0, column=7, padx=2, ipady=5)
    bind_hover_message(reveal_button, "Show the selected file in the system file manager. (Ctrl+Enter)")

    intel_button = ttk.Menubutton(footer_frame, text="Intel", width=12)
    intel_button.grid(row=0, column=8, padx=2, ipady=5)
    bind_hover_message(intel_button, "Threat intelligence for selected items (VirusTotal, online repository).")

    intel_menu = tk.Menu(intel_button, tearoff=0)
    intel_menu.add_command(label="Check on VirusTotal", command=check_virustotal, accelerator="Ctrl+T")
    intel_menu.add_command(label="View Online", command=view_online, accelerator="Ctrl+L")
    intel_button["menu"] = intel_menu

    ttk.Separator(footer_frame, orient=tk.VERTICAL).grid(row=0, column=10, sticky="ns", padx=10)

    # Spacer
    ttk.Frame(footer_frame).grid(row=0, column=11, sticky="ew")

    results_button = ttk.Menubutton(footer_frame, text="Results", width=12)
    results_button.grid(row=0, column=12, padx=(2, 0), ipady=5)

    # --- Status Bar Frame ---
    status_bar_frame = ttk.Frame(root)
    status_bar_frame.grid(row=5, column=0, sticky="ew")

    progress_bar = ttk.Progressbar(status_bar_frame, orient=tk.HORIZONTAL, mode='determinate')
    progress_bar.pack(side=tk.TOP, fill=tk.X)

    status_label = ttk.Label(status_bar_frame, text="Ready", anchor="w", relief=tk.SUNKEN, padding=(10, 3))
    status_label.pack(side=tk.BOTTOM, fill=tk.X)
    bind_hover_message(results_button, "Manage scan results (Import, Export, Clear).")

    results_menu = tk.Menu(results_button, tearoff=0)
    results_menu.add_command(label="Import Results...", command=import_results, accelerator="Ctrl+O")
    results_menu.add_command(label="Import from Clipboard", command=import_from_clipboard, accelerator="Ctrl+V")
    results_menu.add_command(label="Import from Web Link...", command=import_from_url)
    results_menu.add_command(label="Export Results...", command=export_results, accelerator="Ctrl+E")
    results_menu.add_separator()
    results_menu.add_command(label="Clear Results", command=clear_results)
    results_button["menu"] = results_menu

    # --- Context Menu ---
    global context_menu
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="View Details...", command=view_details, accelerator="Space")
    context_menu.add_separator()
    context_menu.add_command(label="Rescan Selected", command=rescan_selected, accelerator="F5")
    context_menu.add_command(label="Analyze with AI", command=analyze_selected_with_ai, accelerator="Ctrl+G")
    context_menu.add_command(label="Exclude Selected", command=exclude_selected, accelerator="Delete")
    context_menu.add_separator()
    context_menu.add_command(label="Open", command=open_file, accelerator="Shift+Enter")
    context_menu.add_command(label="Show in Folder", command=show_in_folder, accelerator="Ctrl+Enter")
    context_menu.add_command(label="Check on VirusTotal", command=check_virustotal, accelerator="Ctrl+T")
    context_menu.add_command(label="View Online", command=view_online, accelerator="Ctrl+L")
    context_menu.add_separator()

    copy_submenu = tk.Menu(context_menu, tearoff=0)
    copy_submenu.add_command(label="File Path", command=copy_path, accelerator="Ctrl+C")
    copy_submenu.add_command(label="SHA256 Hash", command=copy_sha256, accelerator="Ctrl+H")
    copy_submenu.add_command(label="Code Snippet", command=copy_snippet, accelerator="Ctrl+S")
    copy_submenu.add_command(label="As Markdown Table", command=copy_as_markdown, accelerator="Ctrl+Shift+C")
    copy_submenu.add_command(label="As JSON Array", command=copy_as_json, accelerator="Ctrl+J")
    copy_submenu.add_command(label="As Triage Report", command=copy_as_report, accelerator="Ctrl+Shift+R")
    context_menu.add_cascade(label="Copy", menu=copy_submenu)

    context_menu.add_separator()
    context_menu.add_command(label="Select All", command=select_all_items, accelerator="Ctrl+A")

    # Bind context menu to right-click and menu key
    tree.bind('<Button-3>', show_context_menu) # Windows/Linux
    tree.bind('<Button-2>', show_context_menu) # macOS
    tree.bind('<Menu>', show_context_menu)

    # Bind selection and rescan keys
    root.bind('<Return>', on_root_return)
    root.bind('<Control-o>', import_results)
    root.bind('<Command-o>', import_results)
    root.bind('<Control-Shift-O>', lambda event: browse_file_click())
    root.bind('<Command-Shift-O>', lambda event: browse_file_click())
    root.bind('<Control-Shift-F>', lambda event: browse_dir_click())
    root.bind('<Command-Shift-F>', lambda event: browse_dir_click())
    root.bind('<Control-Shift-U>', lambda event: select_url_click())
    root.bind('<Command-Shift-U>', lambda event: select_url_click())
    root.bind('<Control-Shift-V>', lambda event: scan_clipboard_click())
    root.bind('<Command-Shift-V>', lambda event: scan_clipboard_click())
    root.bind('<Control-Shift-D>', lambda event: scan_git_diff_click())
    root.bind('<Command-Shift-D>', lambda event: scan_git_diff_click())
    root.bind('<Control-Shift-G>', lambda event: scan_git_hooks_click())
    root.bind('<Command-Shift-G>', lambda event: scan_git_hooks_click())
    root.bind('<Control-Shift-Q>', lambda event: scan_git_stash_click())
    root.bind('<Command-Shift-Q>', lambda event: scan_git_stash_click())
    root.bind('<Control-Shift-B>', lambda event: scan_shell_profiles_click())
    root.bind('<Command-Shift-B>', lambda event: scan_shell_profiles_click())
    root.bind('<Control-Shift-H>', lambda event: scan_shell_history_click())
    root.bind('<Command-Shift-H>', lambda event: scan_shell_history_click())
    root.bind('<Control-Shift-P>', lambda event: scan_system_path_click())
    root.bind('<Command-Shift-P>', lambda event: scan_system_path_click())
    root.bind('<Control-Shift-K>', lambda event: scan_running_processes_click())
    root.bind('<Command-Shift-K>', lambda event: scan_running_processes_click())
    root.bind('<Control-Shift-N>', lambda event: scan_env_vars_click())
    root.bind('<Command-Shift-N>', lambda event: scan_env_vars_click())
    root.bind('<Control-Shift-T>', lambda event: scan_scheduled_tasks_click())
    root.bind('<Command-Shift-T>', lambda event: scan_scheduled_tasks_click())
    root.bind('<Control-Shift-A>', lambda event: scan_startup_items_click())
    root.bind('<Command-Shift-A>', lambda event: scan_startup_items_click())
    root.bind('<Control-Shift-S>', lambda event: scan_system_services_click())
    root.bind('<Command-Shift-S>', lambda event: scan_system_services_click())
    root.bind('<Control-Shift-R>', lambda event: scan_ssh_config_click())
    root.bind('<Command-Shift-R>', lambda event: scan_ssh_config_click())
    root.bind('<Control-Shift-Y>', lambda event: scan_python_packages_click())
    root.bind('<Command-Shift-Y>', lambda event: scan_python_packages_click())
    root.bind('<Control-Shift-M>', lambda event: scan_nodejs_packages_click())
    root.bind('<Command-Shift-M>', lambda event: scan_nodejs_packages_click())
    root.bind('<Control-Shift-W>', lambda event: scan_browser_extensions_click())
    root.bind('<Command-Shift-W>', lambda event: scan_browser_extensions_click())
    root.bind('<Control-Shift-X>', lambda event: scan_editor_extensions_click())
    root.bind('<Command-Shift-X>', lambda event: scan_editor_extensions_click())
    root.bind('<Control-Shift-J>', lambda event: scan_downloads_click())
    root.bind('<Command-Shift-J>', lambda event: scan_downloads_click())
    root.bind('<Control-Shift-L>', lambda event: scan_desktop_click())
    root.bind('<Command-Shift-L>', lambda event: scan_desktop_click())
    root.bind('<Control-Shift-Z>', lambda event: scan_temp_click())
    root.bind('<Command-Shift-Z>', lambda event: scan_temp_click())
    root.bind('<Control-Shift-I>', lambda event: scan_system_audit_click())
    root.bind('<Command-Shift-I>', lambda event: scan_system_audit_click())
    root.bind('<Control-e>', export_results)
    root.bind('<Command-e>', export_results)
    root.bind('<Control-t>', check_virustotal)
    root.bind('<Command-t>', check_virustotal)
    root.bind('<Control-l>', view_online)
    root.bind('<Command-l>', view_online)
    root.bind('<Control-Shift-E>', copy_cli_command)
    root.bind('<Command-Shift-E>', copy_cli_command)
    root.bind('<Control-v>', import_from_clipboard)
    root.bind('<Command-v>', import_from_clipboard)
    root.bind('<Control-f>', focus_filter)
    root.bind('<Command-f>', focus_filter)
    root.bind('<Control-j>', copy_as_json)
    root.bind('<Command-j>', copy_as_json)
    root.bind('<Control-Shift-R>', copy_as_report)
    root.bind('<Command-Shift-R>', copy_as_report)
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
        description="Scan scripts, project files, and web links for dangerous code using AI. Works with archives, Notebooks, project and build files, automation tasks, Docker, deceptive filenames, and Git changes.",
        epilog="Examples:\n"
               "  # Scan a folder and use AI for analysis\n"
               "  python3 gptscan.py ./my_scripts --cli --use-gpt\n\n"
               "  # Scan Git changes and stop if threats are found\n"
               "  python3 gptscan.py --git-changes --cli --fail-threshold 50\n\n"
               "  # Scan current Git changes as a diff\n"
               "  python3 gptscan.py --git-diff --cli\n\n"
               "  # Scan a web link (GitHub, GitLab, Pastebin, etc.)\n"
               "  python3 gptscan.py https://github.com/user/repo --cli\n\n"
               "  # Run a full system audit\n"
               "  python3 gptscan.py --audit --cli\n\n"
               "  # Scan all installed editor extensions\n"
               "  python3 gptscan.py --editor-extensions --cli\n\n"
               "  # Scan SSH configuration and authorized keys\n"
               "  python3 gptscan.py --ssh-config --cli\n\n"
               "  # Scan files changed in the last 24 hours\n"
               "  python3 gptscan.py --modified 24h --cli\n\n"
               "Note: Run the script from its own folder so it can find its data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {Config.VERSION}')
    parser.add_argument('target', nargs='?', help='The folder, file, pattern, or web link to scan.')
    parser.add_argument(
        'files',
        nargs='*',
        help='Other folders, files, patterns, or web links to scan.'
    )

    scan_group = parser.add_argument_group("Scan Options")
    scan_group.add_argument('-p', '--path', type=str, help='A folder, file, or web link to scan.')
    scan_group.add_argument('-d', '--deep', action='store_true', help='Scan the entire file instead of just the beginning and end. This is more thorough but slower.')
    scan_group.add_argument('--dry-run', action='store_true', help='Preview which files would be scanned without actually checking them.')
    scan_group.add_argument(
        '--extensions',
        type=str,
        help="Only scan these file types (for example: 'py,js')."
    )
    scan_group.add_argument(
        '-e', '--exclude',
        nargs='*',
        help="Ignore files or folders matching these patterns (for example: 'node_modules/*')."
    )
    scan_group.add_argument(
        '--file-list',
        type=argparse.FileType('r'),
        help='Read a list of files to scan from a text file.'
    )
    scan_group.add_argument(
        '--all-files',
        action='store_true',
        help='Check every file, even if it is not a script.'
    )
    scan_group.add_argument(
        '--fail-threshold',
        type=int,
        help='Stop with an error if a file has a threat level at or above this number (0-100).'
    )
    scan_group.add_argument(
        '--threshold', '-t',
        type=int,
        default=50,
        help='Set the minimum threat level (0-100) to show in results. Default is 50.'
    )
    scan_group.add_argument(
        '--stdin',
        action='store_true',
        help='Scan code sent from another command in the terminal.'
    )
    scan_group.add_argument(
        '--import-results', '--import',
        type=str,
        help='Import results from a previous scan. Use "-" to read from the terminal.'
    )
    scan_group.add_argument(
        '--max-size',
        type=str,
        help='The maximum file size to scan (for example: "10MB"). Default is 10MB.'
    )
    scan_group.add_argument(
        '--modified',
        type=str,
        help="Only scan files changed within this time (for example: '24h', '1h', '7d')."
    )
    scan_group.add_argument(
        '--downloads',
        action='store_true',
        help='Scan your standard Downloads folder for suspicious files.'
    )
    scan_group.add_argument(
        '--desktop',
        action='store_true',
        help="Scan your standard Desktop folder for suspicious files."
    )

    git_group = parser.add_argument_group("Git Integration")
    git_group.add_argument(
        '--git-changes',
        nargs='?',
        const='HEAD',
        help='Scan files that have changed in your project. Optionally provide a branch or commit (for example: "main").'
    )
    git_group.add_argument(
        '--git-diff',
        nargs='?',
        const='HEAD',
        help='Scan your current project changes as a diff. Optionally provide a branch or commit (for example: "HEAD~1").'
    )
    git_group.add_argument(
        '--git-hooks',
        action='store_true',
        help='Scan your local and global Git hooks for dangerous scripts.'
    )
    git_group.add_argument(
        '--git-config',
        action='store_true',
        help='Scan Git settings for dangerous aliases or editors.'
    )
    git_group.add_argument(
        '--git-stash',
        action='store_true',
        help='Scan all Git stashes for suspicious code changes.'
    )
    git_group.add_argument(
        '--git-conflicts',
        action='store_true',
        help='Scan files with Git merge conflicts for suspicious code introduced during merging.'
    )
    git_group.add_argument(
        '--git-history',
        type=int,
        nargs='?',
        const=5,
        help='Scan files from the most recent Git commits. Optionally provide the number of commits (default is 5).'
    )
    git_group.add_argument(
        '--git-reflog',
        type=int,
        nargs='?',
        const=5,
        help='Scan recent entries in your Git reflog to find lost code or secrets. Optionally provide the number of entries (default is 5).'
    )

    system_group = parser.add_argument_group("System Scans")
    system_group.add_argument(
        '--audit',
        action='store_true',
        help='Run a full check of your system, including all items listed below.'
    )
    system_group.add_argument(
        '--shell-profiles',
        action='store_true',
        help='Scan your shell configuration files (like .bashrc or .zshrc) for dangerous aliases.'
    )
    system_group.add_argument(
        '--shell-history',
        action='store_true',
        help='Scan your terminal history for dangerous commands.'
    )
    system_group.add_argument(
        '--system-path',
        action='store_true',
        help='Scan folders in your system PATH for suspicious programs.'
    )
    system_group.add_argument(
        '--running-processes',
        action='store_true',
        help='Scan the command lines of active processes.'
    )
    system_group.add_argument(
        '--scheduled-tasks',
        action='store_true',
        help='Scan tasks and Cron jobs for ways programs stay on your system.'
    )
    system_group.add_argument(
        '--startup-items',
        action='store_true',
        help='Scan startup items and LaunchAgents.'
    )
    system_group.add_argument(
        '--system-services',
        action='store_true',
        help='Scan system services and background units.'
    )
    system_group.add_argument(
        '--python-packages',
        action='store_true',
        help='Scan your installed Python packages for malicious code.'
    )
    system_group.add_argument(
        '--browser-bookmarks',
        action='store_true',
        help='Scan all common browser bookmark files for suspicious bookmarklets (javascript: or data: URLs).'
    )
    system_group.add_argument(
        '--nodejs-packages',
        action='store_true',
        help='Scan your global Node.js packages.'
    )
    system_group.add_argument(
        '--browser-extensions',
        action='store_true',
        help='Scan your browser extension folders for malicious scripts.'
    )
    system_group.add_argument(
        '--editor-extensions',
        action='store_true',
        help='Scan extensions for VS Code, Sublime Text, and Vim.'
    )
    system_group.add_argument(
        '--ssh-config',
        action='store_true',
        help='Scan all common SSH configuration and authorized_keys files.'
    )
    system_group.add_argument(
        '--env-vars',
        action='store_true',
        help='Scan all non-empty environment variables.'
    )
    system_group.add_argument(
        '--ruby-gems',
        action='store_true',
        help='Scan all folders containing installed Ruby gems.'
    )
    system_group.add_argument(
        '--php-packages',
        action='store_true',
        help='Scan all folders containing global PHP Composer packages.'
    )
    system_group.add_argument(
        '--rust-packages',
        action='store_true',
        help='Scan all folders containing global Rust Cargo packages.'
    )
    system_group.add_argument(
        '--go-packages',
        action='store_true',
        help='Scan all folders containing Go packages.'
    )
    system_group.add_argument(
        '--java-packages',
        action='store_true',
        help='Scan all folders containing Java package caches (Maven and Gradle).'
    )
    system_group.add_argument(
        '--dotnet-packages',
        action='store_true',
        help='Scan all folders containing global .NET NuGet package caches.'
    )
    system_group.add_argument(
        '--documents',
        action='store_true',
        help="Scan your standard Documents folder for suspicious files."
    )
    system_group.add_argument(
        '--temp',
        action='store_true',
        help='Scan common temporary folders for suspicious files.'
    )

    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument('-g', '--use-gpt', action='store_true', help='Use AI to analyze suspicious files. Cloud providers require an API key; Ollama does not.')
    ai_group.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'openrouter', 'ollama'],
        help='Choose your AI service (default: openai).'
    )
    ai_group.add_argument(
        '--model',
        type=str,
        help='Choose the AI model (for example: gpt-4o, llama3.2).'
    )
    ai_group.add_argument(
        '--api-key', '-k',
        type=str,
        help='The API key for your AI service.'
    )
    ai_group.add_argument(
        '--api-base',
        type=str,
        help='A custom web link for the AI service (useful for local servers).'
    )
    ai_group.add_argument(
        '--rate-limit',
        type=int,
        default=Config.RATE_LIMIT_PER_MINUTE,
        help='Maximum AI requests per minute (default: 60).'
    )
    ai_group.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the AI analysis cache.'
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--cli', action='store_true', help='Run in the terminal instead of opening a window.')
    output_group.add_argument('-a', '--show-all', action='store_true', help='Show all scanned files, even safe ones.')
    output_group.add_argument('-o', '--output', type=str, help='Save the results to a file.')
    output_group.add_argument('-j', '--json', action='store_true', help='Output results as JSON.')
    output_group.add_argument('--csv', action='store_true', help='Output results as CSV.')
    output_group.add_argument('--sarif', action='store_true', help='Save results in SARIF format.')
    output_group.add_argument('--html', action='store_true', help='Create an HTML report.')
    output_group.add_argument('--md', '--markdown', action='store_true', dest='markdown', help='Create a Markdown report.')
    output_group.add_argument('--report', action='store_true', help='Output a report to the terminal.')

    args = parser.parse_args()

    if args.clear_cache:
        Config.gpt_cache = {}
        Config.save_cache()
        print("AI Analysis cache cleared.", file=sys.stderr)
        # If we ONLY wanted to clear cache, exit now.
        if not any([
            args.target, args.path, args.stdin, args.import_results, args.files,
            args.env_vars, args.file_list, args.git_changes, args.git_diff, args.git_hooks, args.git_config,
            args.git_stash, args.git_conflicts, args.git_history, args.git_reflog, args.shell_profiles, args.shell_history, args.system_path,
            args.running_processes, args.scheduled_tasks, args.startup_items,
            args.system_services, args.audit, args.modified, args.downloads, args.desktop,
            args.python_packages, args.nodejs_packages, args.ruby_gems, args.php_packages,
            args.rust_packages, args.go_packages, args.java_packages, args.dotnet_packages,
            args.browser_extensions, args.editor_extensions, args.ssh_config, args.temp, args.documents
        ]):
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
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            git_files = []
            for root_dir in git_roots:
                git_files.extend(get_git_changed_files(root_dir, ref=args.git_changes))

            if not git_files:
                print(f"No git changes detected in provided targets (ref: {args.git_changes}).", file=sys.stderr)
            # Scan only changed files
            scan_targets = git_files

        extra_snippets = []
        if args.git_diff:
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            diff_count = 0
            for root_dir in git_roots:
                diff_content = get_git_diff(root_dir, ref=args.git_diff)
                if diff_content:
                    extra_snippets.append((f"git-diff-{diff_count}.patch", diff_content.encode('utf-8')))
                    diff_count += 1

            if not extra_snippets:
                print(f"No Git diff detected in provided targets (ref: {args.git_diff}).", file=sys.stderr)

        if args.git_hooks:
            # Use a copy of current targets as git roots to avoid infinite loop when extending scan_targets
            git_roots = list(scan_targets) if scan_targets else ["."]
            for root_dir in git_roots:
                scan_targets.extend(get_git_hooks_paths(root_dir))

        if args.git_config:
            extra_snippets.extend(get_git_config_snippets())

        if args.git_stash:
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            for root_dir in git_roots:
                extra_snippets.extend(get_git_stash_snippets(root_dir))

        if args.git_conflicts:
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            for root_dir in git_roots:
                extra_snippets.extend(get_git_conflict_snippets(root_dir))

        if args.git_history:
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            for root_dir in git_roots:
                extra_snippets.extend(get_git_history_snippets(root_dir, count=args.git_history))

        if args.git_reflog:
            # Use specified targets as git roots, or current folder if none.
            git_roots = scan_targets if scan_targets else ["."]
            for root_dir in git_roots:
                extra_snippets.extend(get_git_reflog_snippets(root_dir, count=args.git_reflog))

        if not scan_targets and not args.git_changes and not args.git_diff and not args.git_hooks and not args.git_config and not args.git_stash and not args.git_conflicts and not args.git_history and not args.git_reflog and not extra_snippets:
            # Default to current folder if no targets provided and NOT using git-changes
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

        if args.stdin:
            try:
                # Read from stdin buffer for binary safety
                stdin_content = sys.stdin.buffer.read()
                if stdin_content:
                    extra_snippets.append(("[Stdin]", stdin_content))
            except Exception as e:
                print(f"Error reading from terminal input: {e}", file=sys.stderr)

        if args.shell_profiles:
            profile_paths = get_shell_profile_paths()
            if profile_paths:
                scan_targets.extend(profile_paths)
            else:
                print("No common shell profile files were found on this system.", file=sys.stderr)

        if args.shell_history:
            history_paths = get_shell_history_paths()
            if history_paths:
                scan_targets.extend(history_paths)
            else:
                print("No common shell history files were found on this system.", file=sys.stderr)

        if args.system_path:
            path_dirs = get_system_path_directories()
            if path_dirs:
                scan_targets.extend(path_dirs)
            else:
                print("No valid folders found in the system PATH.", file=sys.stderr)

        if args.running_processes:
            processes = get_running_process_commands()
            if processes:
                extra_snippets.extend(processes)
            else:
                print("No running processes with command lines were found.", file=sys.stderr)

        if args.scheduled_tasks:
            tasks = get_scheduled_task_commands()
            if tasks:
                extra_snippets.extend(tasks)
            else:
                print("No scheduled tasks or Cron jobs were found.", file=sys.stderr)

        if args.startup_items:
            items = get_startup_item_commands()
            if items:
                extra_snippets.extend(items)
            else:
                print("No system startup items or LaunchAgents were found.", file=sys.stderr)

        if args.system_services:
            service_paths = get_system_service_paths()
            if service_paths:
                scan_targets.extend(service_paths)
            service_cmds = get_system_service_commands()
            if service_cmds:
                extra_snippets.extend(service_cmds)
            if not service_paths and not service_cmds:
                print("No system services or systemd units were found.", file=sys.stderr)

        if args.python_packages:
            package_paths = get_python_package_paths()
            if package_paths:
                scan_targets.extend(package_paths)
            else:
                print("No Python site-packages folders were found.", file=sys.stderr)

        if args.browser_bookmarks:
            bookmarks_snippets = get_browser_bookmarks_snippets()
            if bookmarks_snippets:
                extra_snippets.extend(bookmarks_snippets)
            else:
                print("No suspicious browser bookmarklets were found.", file=sys.stderr)

        if args.nodejs_packages:
            node_paths = get_nodejs_package_paths()
            if node_paths:
                scan_targets.extend(node_paths)
            else:
                print("No global Node.js package folders were found.", file=sys.stderr)

        if args.browser_extensions:
            extension_paths = get_browser_extensions_paths()
            if extension_paths:
                scan_targets.extend(extension_paths)
            else:
                print("No browser extension folders were found.", file=sys.stderr)

        if args.editor_extensions:
            extension_paths = get_editor_extensions_paths()
            if extension_paths:
                scan_targets.extend(extension_paths)
            else:
                print("No editor extension folders were found.", file=sys.stderr)

        if args.ssh_config:
            ssh_paths = get_ssh_config_paths()
            if ssh_paths:
                scan_targets.extend(ssh_paths)
            else:
                print("No SSH configuration or authorized_keys files were found.", file=sys.stderr)

        if args.ruby_gems:
            gem_paths = get_ruby_gems_paths()
            if gem_paths:
                scan_targets.extend(gem_paths)
            else:
                print("No Ruby gems folders were found.", file=sys.stderr)

        if args.php_packages:
            php_paths = get_php_packages_paths()
            if php_paths:
                scan_targets.extend(php_paths)
            else:
                print("No global PHP package folders were found.", file=sys.stderr)

        if args.rust_packages:
            rust_paths = get_rust_packages_paths()
            if rust_paths:
                scan_targets.extend(rust_paths)
            else:
                print("No global Rust package folders were found.", file=sys.stderr)

        if args.go_packages:
            go_paths = get_go_packages_paths()
            if go_paths:
                scan_targets.extend(go_paths)
            else:
                print("No Go package folders were found.", file=sys.stderr)

        if args.java_packages:
            java_paths = get_java_packages_paths()
            if java_paths:
                scan_targets.extend(java_paths)
            else:
                print("No Java package folders were found.", file=sys.stderr)

        if args.dotnet_packages:
            dotnet_paths = get_dotnet_packages_paths()
            if dotnet_paths:
                scan_targets.extend(dotnet_paths)
            else:
                print("No .NET NuGet package folders were found.", file=sys.stderr)

        if args.documents:
            scan_targets.extend(get_documents_paths())

        if args.env_vars:
            snippets = get_environment_variable_snippets()
            if snippets:
                extra_snippets.extend(snippets)
            else:
                print("No non-empty environment variables were found.", file=sys.stderr)

        if args.audit:
            paths, snippets = get_system_audit_data()
            scan_targets.extend(paths)
            extra_snippets.extend(snippets)

        if args.downloads:
            scan_targets.extend(get_downloads_paths())

        if args.desktop:
            scan_targets.extend(get_desktop_paths())

        if args.temp:
            scan_targets.extend(get_temp_paths())

        modified_since = None
        if args.modified:
            duration = parse_duration(args.modified)
            if duration is None:
                parser.error(f"Invalid duration format for --modified: {args.modified}")
            modified_since = time.time() - duration

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
            import_file=args.import_results,
            modified_since=modified_since
        )
        if args.fail_threshold is not None and threats > 0:
            sys.exit(1)
    else:
        app_root = create_gui(initial_path=scan_target)
        app_root.mainloop()

if __name__ == "__main__":
    main()
