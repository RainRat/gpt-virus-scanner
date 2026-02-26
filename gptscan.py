import asyncio
import csv
import hashlib
import html
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
import tkinter.scrolledtext as scrolledtext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, Iterable, List, Optional, Tuple, Union

import tkinter as tk
import tkinter.filedialog
import tkinter.font
from tkinter import messagebox
import tkinter.ttk as ttk

# Global GUI variables for thread-safe updates and testing
root: Optional[tk.Tk] = None
textbox: Optional[ttk.Entry] = None
progress_bar: Optional[ttk.Progressbar] = None
status_label: Optional[ttk.Label] = None
deep_var: Optional[tk.BooleanVar] = None
all_var: Optional[tk.BooleanVar] = None
gpt_var: Optional[tk.BooleanVar] = None
dry_var: Optional[tk.BooleanVar] = None
git_var: Optional[tk.BooleanVar] = None
filter_var: Optional[tk.StringVar] = None
tree: Optional[ttk.Treeview] = None
scan_button: Optional[ttk.Button] = None
cancel_button: Optional[ttk.Button] = None
view_button: Optional[ttk.Button] = None
rescan_button: Optional[ttk.Button] = None
context_menu: Optional[tk.Menu] = None
_all_results_cache: List[Tuple[Any, ...]] = []
_last_scan_summary: str = ""


def load_file(filename: str, mode: str = 'single_line') -> Union[str, List[str]]:
    """Reads a file and returns its content.

    Args:
        filename: The path to the file.
        mode: 'single_line' (default) returns the first line.
              'multi_line' returns all lines as a list.

    Returns:
        The file content, or an empty result if the file is missing.
    """
    try:
        with open(filename, 'r') as file:
            if mode == 'single_line':
                return file.readline().strip()
            elif mode == 'multi_line':
                return file.read().splitlines()
    except (FileNotFoundError, PermissionError):
        return [] if mode == 'multi_line' else ''


class Config:
    """Global configuration settings for the scanner."""
    VERSION = "1.4.0"
    SETTINGS_FILE = ".gptscan_settings.json"
    MAXLEN = 1024
    EXPECTED_KEYS = ["administrator", "end-user", "threat-level"]
    MAX_RETRIES = 3
    RATE_LIMIT_PER_MINUTE = 60
    MAX_CONCURRENT_REQUESTS = 5
    gpt_cache: Dict[int, Dict[str, Any]] = {}
    apikey: str = load_file('apikey.txt')
    taskdesc: str = load_file('task.txt')
    GPT_ENABLED: bool = False
    extensions_set: set[str] = set()
    extensions_missing: bool = False
    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_base: Optional[str] = None
    ignore_patterns: List[str] = []
    THRESHOLD: int = 50
    last_path: str = ""
    deep_scan: bool = False
    git_changes_only: bool = False
    show_all_files: bool = False
    use_ai_analysis: bool = False

    DEFAULT_EXTENSIONS = ['.py', '.js', '.bat', '.ps1']

    apikey_missing_message = (
        "No API key found. AI analysis with OpenAI or OpenRouter is disabled, but local scans and Ollama still work."
    )
    task_missing_message = (
        "The 'task.txt' file is missing. AI analysis will be skipped."
    )
    extensions_missing_message = (
        f"The 'extensions.txt' file is missing. Using default types: {', '.join(DEFAULT_EXTENSIONS)}"
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
    def save_settings(cls) -> None:
        """Save persistent GUI settings to a JSON file."""
        settings = {
            "last_path": cls.last_path,
            "deep_scan": cls.deep_scan,
            "git_changes_only": cls.git_changes_only,
            "show_all_files": cls.show_all_files,
            "use_ai_analysis": cls.use_ai_analysis,
            "provider": cls.provider,
            "model_name": cls.model_name,
            "threshold": cls.THRESHOLD,
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
                cls.deep_scan = settings.get("deep_scan", cls.deep_scan)
                cls.git_changes_only = settings.get("git_changes_only", cls.git_changes_only)
                cls.show_all_files = settings.get("show_all_files", cls.show_all_files)
                cls.use_ai_analysis = settings.get("use_ai_analysis", cls.use_ai_analysis)
                cls.provider = settings.get("provider", cls.provider)
                cls.model_name = settings.get("model_name", cls.model_name)
                cls.THRESHOLD = settings.get("threshold", cls.THRESHOLD)
        except Exception as e:
            print(f"Warning: Could not load settings: {e}", file=sys.stderr)

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
        if loaded_ignores:
            cls.ignore_patterns = [
                line.strip() for line in loaded_ignores
                if line.strip() and not line.strip().startswith('#')
            ]

        cls.load_settings()

    @classmethod
    def is_supported_file(cls, file_path: Path, is_explicit: bool = False) -> bool:
        """Check if a file should be scanned based on extension, content or explicit request.

        Args:
            file_path: The path to the file to check.
            is_explicit: Whether the file was specifically requested by the user.

        Returns:
            True if the file matches a known extension, has a script shebang,
            or was explicitly requested.
        """
        if is_explicit:
            return True

        extension = file_path.suffix.lower()
        if extension in cls.extensions_set:
            return True

        # Check shebang for files without recognized extension
        try:
            # Avoid checking very large files for shebangs if they aren't scripts
            # but usually reading just the first line is safe.
            with open(file_path, 'rb') as f:
                header = f.read(2)
                if header == b'#!':
                    # It has a shebang! Read the rest of the first line.
                    first_line = f.readline(126).decode('utf-8', errors='ignore').lower()
                    # Common interpreters for supported or similar script types
                    interpreters = ['python', 'node', 'javascript', 'bash', 'sh', 'zsh', 'perl', 'ruby', 'php', 'pwsh', 'powershell']
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
    """Lazily load and cache the TensorFlow model per process."""

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
    """Lazily instantiate and reuse the asynchronous OpenAI client."""

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
        root.update_idletasks()


def enqueue_ui_update(func: Callable, *args: Any, **kwargs: Any) -> None:
    """Queue a UI update to be processed on the main thread.

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


def bind_hover_message(widget: tk.Widget, message: str) -> None:
    """Bind mouse enter/leave events to update the status label."""
    # Store the previous message to restore it later
    previous_message: List[str] = ["Ready"]

    def on_enter(event):
        if current_cancel_event is None:
            # Save current text, defaulting to Ready if empty
            current_text = status_label.cget("text")
            previous_message[0] = current_text if current_text else "Ready"
            update_status(message)

    def on_leave(event):
        if current_cancel_event is None:
            update_status(previous_message[0])

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def _set_scan_target(path: str) -> None:
    """Update the scan target textbox and set focus to the scan button."""
    if path and textbox:
        textbox.delete(0, tk.END)
        textbox.insert(0, path)
        if scan_button:
            scan_button.focus_set()


def browse_dir_click() -> None:
    """Handle the directory selection dialog and populate the textbox."""
    folder_selected = tkinter.filedialog.askdirectory()
    _set_scan_target(folder_selected)


def browse_file_click() -> None:
    """Handle the file selection dialog and populate the textbox."""
    ext_list = sorted(Config.extensions_set) if Config.extensions_set else Config.DEFAULT_EXTENSIONS
    file_types = [
        ("Script files", ";".join([f"*{e}" for e in ext_list])),
        ("All files", "*.*")
    ]
    file_selected = tkinter.filedialog.askopenfilename(
        title="Select File to Scan",
        filetypes=file_types
    )
    _set_scan_target(file_selected)


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
    cache_key = hash(snippet)
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
    """Wrap strings based on the available column width."""
    if not isinstance(val, str):
        return val

    measure = measure or tkinter.font.Font(font='TkDefaultFont').measure
    words = val.split()
    lines: List[Union[str, List[str]]] = [[]]
    for word in words:
        line = lines[-1] + [word]
        if not lines[-1] or measure(' '.join(line)) < (width - pad):
            lines[-1].append(word)
        else:
            lines[-1] = ' '.join(lines[-1])
            lines.append([word])

    if isinstance(lines[-1], list):
        lines[-1] = ' '.join(lines[-1])

    return '\n'.join(lines)


def get_wrapped_values(tree: ttk.Treeview, values: Iterable[Any], measure: Optional[Callable[[str], int]] = None, col_widths: Optional[List[int]] = None) -> List[Any]:
    """Wrap a list of values to fit the current Treeview column widths."""
    measure = measure or (default_font_measure or tkinter.font.Font(font='TkDefaultFont').measure)
    col_widths = col_widths or [tree.column(cid)['width'] for cid in tree['columns']]

    # Only wrap the first 6 columns, leave the hidden orig_json column as is
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
            values = tree.item(iid)['values']
            new_vals = get_wrapped_values(tree, values, measure=measure, col_widths=col_widths)
            tree.item(iid, values=new_vals)


def get_git_changed_files(path: str = ".") -> List[str]:
    """Get a list of changed files (staged, unstaged, untracked) from git."""
    files = set()

    # Ensure we have a directory for cwd
    if os.path.isfile(path):
        cwd = os.path.dirname(path) or "."
        # If we are looking at a specific file, we only care about that file
        targets = [os.path.basename(path)]
    else:
        cwd = path
        targets = []

    # Changed (staged and unstaged) relative to HEAD
    try:
        cmd = ["git", "diff", "--name-only", "HEAD", "--relative"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # HEAD might not exist, git is missing, or cwd is invalid
        pass

    # Untracked files
    try:
        # Note: git ls-files does not support --relative in all versions,
        # and it returns relative paths by default when run inside a subdirectory.
        cmd = ["git", "ls-files", "--others", "--exclude-standard"] + targets
        output = subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Not a git repo, git is missing, or cwd is invalid
        pass

    return [os.path.join(cwd, f) for f in files if os.path.exists(os.path.join(cwd, f))]


def collect_files(targets: Union[str, List[str]]) -> List[Path]:
    """Collect files from a single path or a list of paths (files or directories).

    Parameters
    ----------
    targets : Union[str, List[str]]
        A single directory path or a list of file/directory paths.

    Returns
    -------
    List[Path]
        A deduplicated list of files to scan.
    """
    if isinstance(targets, str):
        targets = [targets]

    results: List[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_file():
            results.append(p)
        elif p.is_dir():
            results.extend([f for f in p.rglob('*') if f.is_file()])

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


def format_bytes(num: float) -> str:
    """Format a number of bytes into a human-readable string (e.g., KiB, MiB)."""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def get_file_sha256(file_path: Union[str, Path]) -> str:
    """Calculate the SHA256 hash of a file.

    Args:
        file_path: The path to the file.

    Returns:
        The hex digest of the SHA256 hash, or an empty string if calculation fails.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
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


def get_effective_confidence(own_conf_str: Any, gpt_conf_str: Any) -> float:
    """Calculate the effective confidence score, prioritizing GPT over local AI."""
    gpt_val = parse_percent(gpt_conf_str)
    if gpt_val >= 0:
        return gpt_val
    return parse_percent(own_conf_str)


def get_risk_category(conf: float, threshold: int) -> Optional[str]:
    """Categorize a confidence score into 'high', 'medium', or None (no threat)."""
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
        conf = get_effective_confidence(values[1], values[4])
        if conf < Config.THRESHOLD:
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
    # only use the first 6 columns as the 7th is the hidden one itself if updating
    orig_data = list(values[:6])
    orig_json = json.dumps(orig_data)

    wrapped_values = get_wrapped_values(tree, values[:6])
    wrapped_values.append(orig_json)

    # Determine risk level based on confidence scores
    # data format: (path, own_conf, admin, user, gpt_conf, snippet)
    conf = get_effective_confidence(values[1], values[4])
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
        tree["displaycolumns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    else:
        tree["displaycolumns"] = ("path", "own_conf", "snippet")


def set_scanning_state(is_scanning: bool) -> None:
    """Enable or disable controls based on scanning state."""

    if scan_button and cancel_button:
        scan_button.config(state="disabled" if is_scanning else "normal")
        cancel_button.config(state="normal" if is_scanning else "disabled")

    if view_button and rescan_button:
        view_button.config(state="disabled" if is_scanning else "normal")
        rescan_button.config(state="disabled" if is_scanning else "normal")


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
    else:
        update_status("Ready")

    update_tree_columns()

    # Auto-select the first result and focus the tree for immediate keyboard navigation
    if tree:
        items = tree.get_children()
        if items:
            tree.selection_set(items[0])
            tree.focus(items[0])
            tree.focus_set()


def button_click() -> None:
    """Trigger a scan in a background thread using the selected path.

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
    if not scan_path:
        messagebox.showerror("Scan Error", "Please select a file or folder to scan.")
        return

    scan_targets: Union[str, List[str]] = scan_path
    if git_var.get():
        git_files = get_git_changed_files(scan_path)
        if not git_files:
            messagebox.showinfo("Git Scan", "No git changes detected in the selected directory.")
            return
        scan_targets = git_files

    if not dry_var.get() and not os.path.exists('scripts.h5'):
        messagebox.showerror("Scan Error", "Model file scripts.h5 not found.")
        return

    Config.last_path = scan_path
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


def exclude_selected() -> None:
    """Exclude selected files from future scans by adding them to .gptscanignore."""
    if not tree:
        return

    selection = tree.selection()
    if not selection:
        return

    # Ask for confirmation
    if not messagebox.askyesno("Exclude from Scan",
                                f"Are you sure you want to exclude {len(selection)} selected item(s) from future scans?\n"
                                "This will add them to your .gptscanignore file."):
        return

    excluded_paths = []
    for item_id in selection:
        values = _get_item_raw_values(item_id)
        if values:
            path = values[0]
            excluded_paths.append(path)

    if not excluded_paths:
        return

    try:
        # Update .gptscanignore
        ignore_file = Path('.gptscanignore')
        with open(ignore_file, 'a', encoding='utf-8') as f:
            # Add a newline if file is not empty and doesn't end with one
            if ignore_file.exists() and ignore_file.stat().st_size > 0:
                with open(ignore_file, 'r', encoding='utf-8') as fr:
                    content = fr.read()
                    if content and not content.endswith('\n'):
                        f.write('\n')

            for path in excluded_paths:
                # Use relative path if possible for cleaner ignore patterns
                try:
                    rel_path = os.path.relpath(path, os.getcwd())
                    # If it's outside CWD, it might return something like ../...
                    # but match() handles it.
                    f.write(f"{rel_path}\n")
                    if rel_path not in Config.ignore_patterns:
                        Config.ignore_patterns.append(rel_path)
                except ValueError:
                    # Fallback to absolute if relpath fails (e.g. different drives on Windows)
                    f.write(f"{path}\n")
                    if path not in Config.ignore_patterns:
                        Config.ignore_patterns.append(path)

        # Update cache and refresh view
        global _all_results_cache
        for path in excluded_paths:
            # Remove from cache
            _all_results_cache = [v for v in _all_results_cache if v[0] != path]

        _apply_filter()
        update_status(f"Excluded {len(excluded_paths)} file(s).")

    except Exception as e:
        messagebox.showerror("Error", f"Could not update .gptscanignore: {e}")


def iter_windows(fh, size: int, deep_scan: bool, maxlen: Optional[int] = None) -> Generator[Tuple[int, bytes], None, None]:
    """Yield file chunks for scanning."""
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
) -> Generator[Tuple[str, Tuple[Any, ...]], None, None]:
    """Scan files for malicious content and optionally request GPT analysis.

    Parameters
    ----------
    scan_targets : Union[str, List[str]]
        Directory path or list of file/directory paths to search.
    deep_scan : bool
        Whether to scan overlapping 1024-byte windows beyond the first block.
    show_all : bool
        Whether to yield all scanned files regardless of confidence threshold.
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

    Yields
    ------
    Generator[Tuple[str, Any], None, None]
        Tuples indicating events:
        - ('progress', (current: int, total: int, status: Optional[str]))
        - ('result', (path: str, own_conf: str, admin: str, user: str, gpt: str, snippet: str))
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

    # Identify which files were explicitly passed as targets
    if isinstance(scan_targets, (str, Path)):
        explicit_targets = {Path(scan_targets)}
    else:
        explicit_targets = {Path(t) for t in scan_targets}
    explicit_files = {f for f in explicit_targets if f.is_file()}

    file_list = collect_files(scan_targets)

    if exclude_patterns:
        file_list = [
            f for f in file_list
            if not any(f.match(p) for p in exclude_patterns)
        ]

    total_progress = len(file_list)
    progress_count = 0
    total_bytes_scanned = 0
    start_time = time.perf_counter()
    yield ('progress', (progress_count, total_progress, "Collecting files..."))

    gpt_requests: List[Dict[str, Any]] = []

    for index, file_path in enumerate(file_list):
        if cancel_event.is_set():
            break

        progress_count = index + 1
        yield ('progress', (progress_count, total_progress, f"Scanning: {file_path.name}"))

        is_explicit = file_path in explicit_files
        if Config.is_supported_file(file_path, is_explicit=is_explicit):
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
                        )
                    )

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
                    )
                )
            else:
                print(file_path, file=sys.stderr)
                if file_size is not None:
                    maxconf = -1.0
                    max_window_bytes = b""
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
                            )
                        )
                    elif maxconf >= 0:
                        percent = f"{maxconf:.0%}"
                        snippet = ''.join(map(chr, max_window_bytes)).strip()
                        cleaned_snippet = ''.join([s for s in snippet.strip().splitlines(True) if s.strip()])
                        threshold_val = Config.THRESHOLD / 100.0
                        if maxconf > threshold_val and use_gpt and Config.GPT_ENABLED:
                            gpt_requests.append(
                                {
                                    "path": str(file_path),
                                    "percent": percent,
                                    "snippet": snippet,
                                    "cleaned_snippet": cleaned_snippet,
                                }
                            )
                        elif maxconf > threshold_val or show_all:
                            yield (
                                'result',
                                (
                                    str(file_path),
                                    percent,
                                    '',
                                    '',
                                    '',
                                    cleaned_snippet,
                                )
                            )

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
                json_data = await async_handle_gpt_response(
                    request["snippet"],
                    Config.taskdesc,
                    rate_limiter=rate_limiter,
                    semaphore=semaphore,
                    wait_callback=wait_notifier,
                )
                return request, json_data

            tasks = [asyncio.create_task(run_request(request)) for request in requests]
            results: List[Tuple[Dict[str, Any], Optional[Dict]]] = []
            for completed in asyncio.as_completed(tasks):
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
                admin_desc = 'JSON Parse Error'
                enduser_desc = 'JSON Parse Error'
                chatgpt_conf_percent = 'JSON Parse Error'
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
                )
            )

    end_time = time.perf_counter()
    yield ('summary', (len(file_list), total_bytes_scanned, end_time - start_time))


def run_scan(
    scan_targets: Union[str, List[str]],
    deep_scan: bool,
    show_all: bool,
    use_gpt: bool,
    cancel_event: threading.Event,
    rate_limit: int = Config.RATE_LIMIT_PER_MINUTE,
    dry_run: bool = False,
    exclude_patterns: Optional[List[str]] = None,
) -> None:
    """Consume scan events and forward them to the UI thread.

    Parameters
    ----------
    scan_targets : Union[str, List[str]]
        Directory path or list of files to scan.
    deep_scan : bool
        Whether to evaluate all 1024-byte windows.
    show_all : bool
        Whether to display all results regardless of confidence.
    use_gpt : bool
        Whether to enrich suspicious files with GPT output.
    rate_limit : int
        Maximum allowed GPT requests per minute.
    dry_run : bool
        Whether to simulate the scan.
    """
    last_total: Optional[int] = 0
    threats_found = 0
    high_risk_found = 0
    medium_risk_found = 0
    metrics: Dict[str, Any] = {}
    current_scanned = 0

    try:
        for event_type, data in scan_files(
            scan_targets,
            deep_scan,
            show_all,
            use_gpt,
            cancel_event,
            rate_limit=rate_limit,
            max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        ):
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
                status_text = f"{status} ({current}/{total}){threat_suffix}" if status else f"Scanning: {current}/{total}{threat_suffix}"
                print(status_text, file=sys.stderr)
                enqueue_ui_update(update_status, status_text)
            elif event_type == 'result':
                if cancel_event.is_set():
                    continue
                # data format: (path, own_conf, admin, user, gpt_conf, snippet)
                conf = get_effective_confidence(data[1], data[4])
                risk = get_risk_category(conf, Config.THRESHOLD)

                if risk == 'high':
                    threats_found += 1
                    high_risk_found += 1
                elif risk == 'medium':
                    threats_found += 1
                    medium_risk_found += 1

                if not cancel_event.is_set():
                    enqueue_ui_update(insert_tree_row, data)
            elif event_type == 'summary':
                total_files, total_bytes, elapsed_time = data
                metrics['total_bytes'] = total_bytes
                metrics['elapsed_time'] = elapsed_time
    finally:
        enqueue_ui_update(
            finish_scan_state,
            current_scanned,
            threats_found,
            metrics.get('total_bytes'),
            metrics.get('elapsed_time'),
            high_risk_found,
            medium_risk_found
        )


def run_rescan(
    paths: List[str],
    item_map: Dict[str, str],
    settings: Dict[str, Any],
    cancel_event: threading.Event
) -> None:
    """Perform background scan for specific paths and update existing UI rows."""
    threats_found = 0
    high_risk_found = 0
    medium_risk_found = 0
    metrics: Dict[str, Any] = {}
    current_scanned = 0
    last_total: Optional[int] = 0

    try:
        for event_type, data in scan_files(
            paths,
            settings['deep'],
            show_all=True,  # Always show results for rescan to update rows
            use_gpt=settings['gpt'],
            cancel_event=cancel_event,
            rate_limit=Config.RATE_LIMIT_PER_MINUTE,
            max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
            dry_run=settings['dry'],
            exclude_patterns=None,  # Already selected, don't re-exclude
        ):
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
                status_text = f"{status} ({current}/{total}){threat_suffix}" if status else f"Rescanning: {current}/{total}{threat_suffix}"
                enqueue_ui_update(update_status, status_text)
            elif event_type == 'result':
                if cancel_event.is_set():
                    continue
                path = data[0]
                item_id = item_map.get(path)
                if item_id:
                    conf = get_effective_confidence(data[1], data[4])
                    risk = get_risk_category(conf, Config.THRESHOLD)
                    if risk == 'high':
                        threats_found += 1
                        high_risk_found += 1
                    elif risk == 'medium':
                        threats_found += 1
                        medium_risk_found += 1
                    enqueue_ui_update(update_tree_row, item_id, data)
            elif event_type == 'summary':
                total_files, total_bytes, elapsed_time = data
                metrics['total_bytes'] = total_bytes
                metrics['elapsed_time'] = elapsed_time
    finally:
        enqueue_ui_update(
            finish_scan_state,
            current_scanned,
            threats_found,
            metrics.get('total_bytes'),
            metrics.get('elapsed_time'),
            high_risk_found,
            medium_risk_found
        )


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
        # Convert confidence strings to levels
        level = "note"
        conf = get_effective_confidence(r.get("own_conf", ""), r.get("gpt_conf", ""))
        if conf > 80:
            level = "error"
        elif conf > Config.THRESHOLD:
            level = "warning"

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

        conf_val = get_effective_confidence(own_conf, gpt_conf)

        row_class = ""
        if conf_val > 80:
            row_class = "high-risk"
        elif conf_val > Config.THRESHOLD:
            row_class = "medium-risk"

        rows.append(f"""
        <tr class="{row_class}">
            <td>{html.escape(path)}</td>
            <td>{html.escape(gpt_conf or own_conf)}</td>
            <td>
                <strong>Admin:</strong> {html.escape(admin)}<br>
                <strong>User:</strong> {html.escape(user)}
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
                <th style="width: 10%">Confidence</th>
                <th style="width: 30%">Analysis</th>
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
        "| Path | Confidence | Analysis | Snippet |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        path = r.get("path", "").replace("|", "\\|")
        own_conf = r.get("own_conf", "")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        snippet = r.get("snippet", "")

        conf_str = gpt_conf or own_conf
        analysis_parts = []
        if admin:
            analysis_parts.append(f"**Admin:** {admin.replace('|', '\\|')}")
        if user:
            analysis_parts.append(f"**User:** {user.replace('|', '\\|')}")
        analysis = "<br>".join(analysis_parts)

        # Clean up snippet for markdown table (one line, escaped)
        clean_snippet = snippet.replace("\n", " ").replace("|", "\\|")
        if len(clean_snippet) > 100:
            clean_snippet = clean_snippet[:97] + "..."

        lines.append(f"| {path} | {conf_str} | {analysis} | `{clean_snippet}` |")

    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")

    for r in results:
        path = r.get("path", "")
        own_conf = r.get("own_conf", "")
        gpt_conf = r.get("gpt_conf", "")
        admin = r.get("admin_desc", "")
        user = r.get("end-user_desc", "")
        snippet = r.get("snippet", "")

        lines.append(f"### File: `{path}`")
        lines.append(f"- **Local Confidence:** {own_conf}")
        if gpt_conf:
            lines.append(f"- **AI Confidence:** {gpt_conf}")
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
        lines.append(f"```{lang}")
        lines.append(snippet)
        lines.append("```")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def run_cli(targets: Union[str, List[str]], deep: bool, show_all: bool, use_gpt: bool, rate_limit: int, output_format: str = 'csv', dry_run: bool = False, exclude_patterns: Optional[List[str]] = None, fail_threshold: Optional[int] = None, output_file: Optional[str] = None) -> int:
    """Run scans and stream results to stdout or a file.

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
        Confidence threshold to trigger a failure count.
    output_file : str, optional
        Path to a file where results should be saved.
    """
    keys = ["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"]

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

    for event_type, data in scan_files(
        targets,
        deep,
        show_all,
        use_gpt,
        cancel_event,
        rate_limit=rate_limit,
        max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
        dry_run=dry_run,
        exclude_patterns=exclude_patterns,
    ):
        if event_type == 'result':
            # data format: (path, own_conf, admin, user, gpt_conf, snippet)
            conf = get_effective_confidence(data[1], data[4])

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
            elif output_format in ('sarif', 'html', 'markdown'):
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

    if output_file:
        out_stream.close()

    return threats_found


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

    file_path = tkinter.filedialog.askopenfilename(
        filetypes=[
            ("All supported formats", "*.json;*.jsonl;*.ndjson;*.csv;*.sarif"),
            ("JSON files", "*.json;*.jsonl;*.ndjson"),
            ("SARIF files", "*.sarif"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ],
        title="Import Scan Results",
    )
    if not file_path:
        return

    try:
        data_to_import = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ('.json', '.jsonl', '.ndjson', '.sarif'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("File is empty.")

                if content.startswith('['):
                    # Standard JSON list
                    data_to_import = json.loads(content)
                elif ext == '.sarif' or (content.startswith('{') and '"runs"' in content):
                    # SARIF format
                    sarif_data = json.loads(content)
                    data_to_import = []
                    for run in sarif_data.get("runs", []):
                        for result in run.get("results", []):
                            # Extract properties
                            props = result.get("properties", {})

                            mapped = {
                                "path": "",
                                "own_conf": props.get("own_conf", ""),
                                "admin_desc": props.get("admin_desc") or result.get("message", {}).get("text", ""),
                                "end-user_desc": props.get("end-user_desc", ""),
                                "gpt_conf": props.get("gpt_conf", ""),
                                "snippet": props.get("snippet", "")
                            }
                            # Extract path
                            locations = result.get("locations", [])
                            if locations:
                                uri = locations[0].get("physicalLocation", {}).get("artifactLocation", {}).get("uri", "")
                                mapped["path"] = uri.replace("/", os.sep)

                            data_to_import.append(mapped)
                else:
                    # NDJSON (newline-delimited JSON)
                    data_to_import = [json.loads(line) for line in content.splitlines() if line.strip()]
        elif ext == '.csv':
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data_to_import = list(reader)
        else:
            messagebox.showerror("Import Error", f"Unsupported file extension: {ext}")
            return

        if not data_to_import:
            messagebox.showwarning("Import Warning", "No data found in the selected file.")
            return

        # Clear existing results
        clear_results()

        columns = tree["columns"]
        count = 0
        for item in data_to_import:
            # item is a dict from JSON or csv.DictReader
            # Map item keys back to the expected column order
            values = []
            for col in columns:
                # Try exact match first
                val = item.get(col)
                if val is None:
                    # Try to map some known alternatives or header names
                    if col == "path":
                        val = item.get("File Path")
                    elif col == "own_conf":
                        val = item.get("Local Conf.")
                    elif col == "admin_desc":
                        val = item.get("Admin Notes")
                    elif col == "end-user_desc":
                        val = item.get("User Notes")
                    elif col == "gpt_conf":
                        val = item.get("AI Conf.")
                    elif col == "snippet":
                        val = item.get("Snippet")

                values.append(val if val is not None else "")

            insert_tree_row(tuple(values))
            count += 1

        msg = f"Imported {count} results from {os.path.basename(file_path)}"
        global _last_scan_summary
        _last_scan_summary = msg
        update_status(msg)
        update_tree_columns()

        # Auto-select the first result and focus the tree for immediate keyboard navigation
        items = tree.get_children()
        if items:
            tree.selection_set(items[0])
            tree.focus(items[0])
            tree.focus_set()

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not load results:\n{err}")


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


def _get_tree_results_as_dicts(item_ids: Iterable[str]) -> List[Dict[str, Any]]:
    """Extract raw results from the given Treeview item IDs as a list of dictionaries."""
    if not tree:
        return []

    columns = tree["columns"][:6]
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

    file_path = tkinter.filedialog.asksaveasfilename(
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
                columns = tree["columns"][:6]
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

    # Try to return raw values from the hidden column (index 6) if available
    if len(values) > 6 and values[6]:
        try:
            return json.loads(values[6])
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: unwrap display newlines by replacing them with spaces
    return [str(v).replace('\n', ' ') for v in values[:6]]


def _get_selected_row_values() -> Optional[List[Any]]:
    """Retrieve raw values from the currently selected Treeview row."""
    if not tree:
        return None
    selection = tree.selection()
    if not selection:
        return None
    return _get_item_raw_values(selection[0])


def view_details(event: Optional[tk.Event] = None, item_id: Optional[str] = None) -> None:
    """Open a detailed view of the selected scan result."""
    if item_id is None:
        selection = tree.selection()
        if not selection:
            return
        item_id = selection[0]

    values = _get_item_raw_values(item_id)
    if not values:
        return

    # values: (path, own_conf, admin, user, gpt_conf, snippet)
    path = values[0]

    details_win = tk.Toplevel(root)
    details_win.title(f"Result Details - {os.path.basename(path)}")
    details_win.geometry("700x650")
    details_win.minsize(500, 450)

    # Make it modal-ish but not blocking
    details_win.transient(root)

    main_frame = ttk.Frame(details_win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header: Path and Confidence
    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill=tk.X, pady=(0, 5))

    ttk.Label(header_frame, text="File Path:", font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, sticky="w")
    path_entry = ttk.Entry(header_frame)
    path_entry.grid(row=0, column=1, sticky="ew", padx=5)
    header_frame.columnconfigure(1, weight=1)

    def copy_path_btn():
        root.clipboard_clear()
        root.clipboard_append(path_entry.get())
        messagebox.showinfo("Copied", "File path copied to clipboard.")

    ttk.Button(header_frame, text="Copy Path", width=12, command=copy_path_btn).grid(row=0, column=2, padx=2)
    ttk.Button(header_frame, text="Show in Folder", width=15, command=lambda: show_in_folder(path_entry.get())).grid(row=0, column=3, padx=2)

    conf_frame = ttk.Frame(header_frame)
    conf_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))

    ttk.Label(conf_frame, text="Local Confidence:").grid(row=0, column=0, sticky="w")
    own_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))
    own_conf_label.grid(row=0, column=1, sticky="w", padx=(5, 20))

    ai_conf_prefix = ttk.Label(conf_frame, text="AI Confidence:")
    gpt_conf_label = ttk.Label(conf_frame, font=('TkDefaultFont', 9, 'bold'))

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

    # Footer buttons
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=(10, 0))

    def on_analyze_now():
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
        text = f"Path: {path}\nLocal Conf: {own_conf}\n"
        if gpt_conf:
            text += f"AI Conf: {gpt_conf}\n"
        if analysis_frame.winfo_viewable():
            if admin_label.winfo_viewable():
                text += f"\nAdmin Notes:\n{admin_text.get('1.0', tk.END).strip()}\n"
            if user_label.winfo_viewable():
                text += f"\nUser Notes:\n{user_text.get('1.0', tk.END).strip()}\n"
        text += f"\nSnippet:\n{snippet_text.get('1.0', tk.END).strip()}"
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("Copied", "Detailed analysis copied to clipboard.")

    ttk.Button(btn_frame, text="Open File", command=lambda: open_file(path_entry.get())).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Copy Analysis", command=copy_analysis).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="VirusTotal", command=lambda: check_virustotal(path_entry.get())).pack(side=tk.LEFT, padx=5)

    analyze_btn = ttk.Button(btn_frame, text="Analyze with AI", command=on_analyze_now)
    analyze_btn.pack(side=tk.LEFT, padx=5)
    if not Config.GPT_ENABLED:
        analyze_btn.config(state='disabled')

    ttk.Button(btn_frame, text="Close", command=details_win.destroy).pack(side=tk.RIGHT, padx=5)

    # Navigation buttons
    nav_frame = ttk.Frame(main_frame)
    nav_frame.pack(fill=tk.X, pady=(10, 0))
    current_item_id = item_id

    def refresh_content(new_id):
        nonlocal current_item_id
        current_item_id = new_id
        vals = _get_item_raw_values(new_id)
        if not vals: return
        path, own_conf, admin, user, gpt_conf, snippet = vals[:6]

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

        snippet_text.config(state='normal')
        snippet_text.delete('1.0', tk.END)
        snippet_text.insert(tk.END, snippet)
        snippet_text.config(state='disabled')

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

    prev_btn = ttk.Button(nav_frame, text="< Previous", command=on_prev)
    prev_btn.pack(side=tk.LEFT, padx=5)

    count_label = ttk.Label(nav_frame, text="")
    count_label.pack(side=tk.LEFT, padx=10)

    next_btn = ttk.Button(nav_frame, text="Next >", command=on_next)
    next_btn.pack(side=tk.LEFT, padx=5)
    details_win.bind('<Left>', lambda e: on_prev())
    details_win.bind('<Right>', lambda e: on_next())
    details_win.bind('<Escape>', lambda e: details_win.destroy())
    refresh_content(item_id)


def open_file(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Open the selected or specified file in the system's default application."""
    if isinstance(event_or_path, str):
        file_path = event_or_path
    else:
        values = _get_selected_row_values()
        if not values:
            return
        file_path = str(values[0])
    if os.path.exists(file_path):
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
    else:
        messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")


def copy_path() -> None:
    """Copy the selected file's path to the clipboard."""
    values = _get_selected_row_values()
    if not values:
        return
    file_path = str(values[0])
    tree.clipboard_clear()
    tree.clipboard_append(file_path)


def copy_sha256() -> None:
    """Calculate and copy the selected file's SHA256 hash to the clipboard."""
    values = _get_selected_row_values()
    if not values:
        return
    file_path = str(values[0])
    h = get_file_sha256(file_path)
    if h:
        tree.clipboard_clear()
        tree.clipboard_append(h)
        update_status(f"SHA256 copied: {h[:8]}...")
    else:
        messagebox.showwarning("Error", "Could not calculate file hash.")


def check_virustotal(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Check the selected or specified file's hash on VirusTotal."""
    if isinstance(event_or_path, str):
        file_path = event_or_path
    else:
        values = _get_selected_row_values()
        if not values:
            return
        file_path = str(values[0])

    if not os.path.exists(file_path):
        messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")
        return

    h = get_file_sha256(file_path)
    if h:
        url = f"https://www.virustotal.com/gui/file/{h}"
        webbrowser.open(url)
        update_status(f"Opening VirusTotal for {os.path.basename(file_path)}...")
    else:
        messagebox.showwarning("Error", "Could not calculate file hash.")


def copy_snippet() -> None:
    """Copy the selected row's code snippet to the clipboard."""
    values = _get_selected_row_values()
    if not values:
        return
    # Snippet is the last column in the 6-column data
    snippet = str(values[5])
    tree.clipboard_clear()
    tree.clipboard_append(snippet)


def copy_as_markdown() -> None:
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


def show_in_folder(event_or_path: Union[tk.Event, str, None] = None) -> None:
    """Reveal the selected or specified file in the system file manager."""
    if isinstance(event_or_path, str):
        file_path = event_or_path
    else:
        values = _get_selected_row_values()
        if not values:
            return
        file_path = str(values[0])
    if os.path.exists(file_path):
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
    else:
        messagebox.showwarning("File Not Found", f"The file '{file_path}' could not be located.")


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
    if not tree or not view_button or not rescan_button:
        return

    has_selection = bool(tree.selection())
    view_button.config(state="normal" if has_selection else "disabled")
    rescan_button.config(state="normal" if has_selection else "disabled")


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
    global root, textbox, progress_bar, status_label, deep_var, all_var, gpt_var, dry_var, git_var, filter_var, tree, scan_button, cancel_button, view_button, rescan_button, default_font_measure

    root = tk.Tk()
    root.geometry("1000x600")
    root.title("GPT Virus Scanner")
    default_font_measure = tkinter.font.Font(font='TkDefaultFont').measure

    # --- Menu Bar ---
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Import Results...", command=import_results)
    file_menu.add_command(label="Export Results...", command=export_results)
    file_menu.add_separator()
    file_menu.add_command(label="Clear Results", command=clear_results)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", f"GPT Virus Scanner v{Config.VERSION}\nA security tool for scanning scripts using AI."))
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
    textbox = ttk.Entry(input_frame)
    path_to_use = initial_path if initial_path else (Config.last_path if Config.last_path else os.getcwd())
    textbox.insert(0, path_to_use)
    textbox.select_range(0, tk.END)
    textbox.grid(row=0, column=1, sticky="ew", padx=5)
    textbox.bind('<Return>', lambda event: button_click())
    textbox.focus_set()

    def on_root_return(event):
        # Trigger scan if focus is not on results tree or textbox (which has its own binding)
        focused = root.focus_get()
        if str(focused) not in (str(tree), str(textbox)):
            button_click()

    root.bind('<Return>', on_root_return)
    root.bind('<Escape>', lambda event: cancel_scan())
    select_file_btn = ttk.Button(input_frame, text="File...", command=browse_file_click)
    select_file_btn.grid(row=0, column=2, sticky="e", padx=(5, 0))
    bind_hover_message(select_file_btn, "Select a single script file to scan.")

    select_dir_btn = ttk.Button(input_frame, text="Folder...", command=browse_dir_click)
    select_dir_btn.grid(row=0, column=3, sticky="e", padx=(5, 0))
    bind_hover_message(select_dir_btn, "Select a directory to scan.")

    scan_button = ttk.Button(input_frame, text="Scan now", command=button_click, default='active')
    scan_button.grid(row=0, column=4, sticky="e", padx=(5, 0))
    bind_hover_message(scan_button, "Start the scan.")

    cancel_button = ttk.Button(input_frame, text="Cancel", command=cancel_scan, state="disabled")
    cancel_button.grid(row=0, column=5, sticky="e", padx=(5, 0))
    bind_hover_message(cancel_button, "Stop the current scan.")

    # --- Settings Container ---
    settings_frame = ttk.Frame(root)
    settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

    # --- Options Frame ---
    options_frame = ttk.LabelFrame(settings_frame, text="Scan Options")
    options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

    gpt_var = tk.BooleanVar(value=Config.use_ai_analysis)

    git_var = tk.BooleanVar(value=Config.git_changes_only)
    git_checkbox = ttk.Checkbutton(options_frame, text="Git changes only", variable=git_var)
    git_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(git_checkbox, "Only scan files that have been modified or are untracked in Git.")

    deep_var = tk.BooleanVar(value=Config.deep_scan)
    deep_checkbox = ttk.Checkbutton(options_frame, text="Deep scan", variable=deep_var)
    deep_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(deep_checkbox, "Scan the whole file. This is slower but more thorough. Normally, the scanner only checks the beginning and end.")

    all_var = tk.BooleanVar(value=Config.show_all_files)
    all_checkbox = ttk.Checkbutton(options_frame, text="Show all files", variable=all_var, command=_apply_filter)
    all_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(all_checkbox, "Display all scanned files, including safe ones.")

    dry_var = tk.BooleanVar()
    dry_checkbox = ttk.Checkbutton(options_frame, text="Dry Run", variable=dry_var)
    dry_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(dry_checkbox, "Simulate the scan process without running checks.")

    threshold_row = ttk.Frame(options_frame)
    threshold_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    ttk.Label(threshold_row, text="Min. Threat Level (%):").pack(side=tk.LEFT)

    def on_threshold_change():
        try:
            val = int(threshold_spin.get())
            Config.THRESHOLD = max(0, min(100, val))
            _apply_filter()
        except ValueError:
            pass

    threshold_spin = tk.Spinbox(threshold_row, from_=0, to=100, width=5, command=on_threshold_change)
    threshold_spin.delete(0, tk.END)
    threshold_spin.insert(0, str(Config.THRESHOLD))
    threshold_spin.pack(side=tk.LEFT, padx=5)
    threshold_spin.bind('<KeyRelease>', lambda e: on_threshold_change())
    bind_hover_message(threshold_spin, "Files with a threat score lower than this will be ignored. Set this higher to see only the most dangerous files.")

    # --- Provider Frame ---
    provider_frame = ttk.LabelFrame(settings_frame, text="AI Analysis")
    provider_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

    def toggle_ai_controls():
        enabled = gpt_var.get()
        if enabled:
            provider_combo.config(state="readonly")
            model_combo.config(state="normal")
        else:
            provider_combo.config(state="disabled")
            model_combo.config(state="disabled")
        update_tree_columns()

    gpt_checkbox = ttk.Checkbutton(provider_frame, text="Use AI Analysis", variable=gpt_var, command=toggle_ai_controls)
    gpt_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(gpt_checkbox, "Use AI to analyze suspicious code and explain what it does.")

    if not Config.GPT_ENABLED:
        gpt_var.set(False)
        gpt_checkbox.config(state="disabled")
        messagebox.showwarning("AI Analysis Disabled",
                                       "task.txt not found. AI Analysis is disabled.")

    provider_row = ttk.Frame(provider_frame)
    provider_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
    ttk.Label(provider_row, text="Provider:", width=10).pack(side=tk.LEFT)
    provider_var = tk.StringVar(value=Config.provider)
    provider_combo = ttk.Combobox(provider_row, textvariable=provider_var, values=["openai", "openrouter", "ollama"], state="readonly", width=12)
    provider_combo.pack(side=tk.LEFT, padx=5)

    model_row = ttk.Frame(provider_frame)
    model_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
    ttk.Label(model_row, text="Model:", width=10).pack(side=tk.LEFT)
    model_var = tk.StringVar(value=Config.model_name)
    model_combo = ttk.Combobox(model_row, textvariable=model_var, width=20)
    model_combo.pack(side=tk.LEFT, padx=5)

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

    ttk.Label(filter_frame, text="Filter results:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    filter_var = tk.StringVar()
    filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
    filter_entry.grid(row=0, column=1, sticky="ew")
    filter_entry.bind('<KeyRelease>', _apply_filter)
    bind_hover_message(filter_entry, "Search results by any column (path, confidence, analysis, snippet).")

    def clear_filter():
        filter_var.set("")
        _apply_filter()

    clear_filter_btn = ttk.Button(filter_frame, text="Clear", width=8, command=clear_filter)
    clear_filter_btn.grid(row=0, column=2, padx=(5, 0))
    bind_hover_message(clear_filter_btn, "Clear the filter and show all results.")

    # --- Treeview ---
    style = ttk.Style(root)
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
    tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("path", width=150, stretch=tk.YES, anchor="w")
    tree.column("own_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("admin_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("end-user_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("gpt_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("snippet", width=150, stretch=tk.YES, anchor="w")
    tree.column("orig_json", width=0, stretch=tk.NO) # Hidden column for raw data
    root.after(0, process_ui_queue)

    tree.heading("#0", text="")
    tree.heading("path", text="File Path", command=lambda: sort_column(tree, "path", False))
    tree.heading("own_conf", text="Local Conf.",
                 command=lambda: sort_column(tree, "own_conf", False))
    tree.heading("admin_desc", text="Admin Notes",
                 command=lambda: sort_column(tree, "admin_desc", False))
    tree.heading("end-user_desc", text="User Notes",
                 command=lambda: sort_column(tree, "end-user_desc", False))
    tree.heading("gpt_conf", text="AI Conf.",
                 command=lambda: sort_column(tree, "gpt_conf", False))
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

    view_button = ttk.Button(footer_frame, text="View Details...", command=view_details)
    view_button.grid(row=0, column=1, padx=2)
    bind_hover_message(view_button, "Show full analysis and code for the selected result.")

    rescan_button = ttk.Button(footer_frame, text="Rescan Selected", command=rescan_selected)
    rescan_button.grid(row=0, column=2, padx=2)
    bind_hover_message(rescan_button, "Re-scan the currently selected items.")

    import_button = ttk.Button(footer_frame, text="Import Results...", command=import_results)
    import_button.grid(row=0, column=3, padx=2)
    bind_hover_message(import_button, "Load results from a JSON or CSV file.")

    export_button = ttk.Button(footer_frame, text="Export Results...", command=export_results)
    export_button.grid(row=0, column=4, padx=2)
    bind_hover_message(export_button, "Save results to CSV, HTML, JSON, or SARIF.")

    clear_button = ttk.Button(footer_frame, text="Clear Results", command=clear_results)
    clear_button.grid(row=0, column=5, padx=(2, 0))
    bind_hover_message(clear_button, "Clear all results from the list.")

    # --- Context Menu ---
    global context_menu
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="View Details...", command=view_details)
    context_menu.add_separator()
    context_menu.add_command(label="Rescan Selected", command=rescan_selected)
    context_menu.add_command(label="Exclude from future scans", command=exclude_selected)
    context_menu.add_separator()
    context_menu.add_command(label="Select All", command=select_all_items)
    context_menu.add_separator()
    context_menu.add_command(label="Open File", command=open_file)
    context_menu.add_command(label="Show in Folder", command=show_in_folder)
    context_menu.add_separator()
    context_menu.add_command(label="Copy File Path", command=copy_path)
    context_menu.add_command(label="Copy SHA256", command=copy_sha256)
    context_menu.add_command(label="Check on VirusTotal", command=check_virustotal)
    context_menu.add_command(label="Copy Snippet", command=copy_snippet)
    context_menu.add_command(label="Copy as Markdown", command=copy_as_markdown)

    # Bind context menu to right-click and menu key
    tree.bind('<Button-3>', show_context_menu) # Windows/Linux
    tree.bind('<Button-2>', show_context_menu) # macOS
    tree.bind('<Menu>', show_context_menu)

    # Bind selection and rescan keys
    root.bind('<Control-f>', lambda e: filter_entry.focus_set())
    root.bind('<Command-f>', lambda e: filter_entry.focus_set())
    tree.bind('<<TreeviewSelect>>', update_button_states)
    tree.bind('<Control-a>', select_all_items)
    tree.bind('<Command-a>', select_all_items)
    tree.bind('<space>', view_details)
    tree.bind('<F5>', lambda event: rescan_selected())
    tree.bind('r', lambda event: rescan_selected())

    def on_close():
        Config.last_path = textbox.get()
        Config.deep_scan = deep_var.get()
        Config.git_changes_only = git_var.get()
        Config.show_all_files = all_var.get()
        Config.use_ai_analysis = gpt_var.get()
        Config.provider = provider_var.get()
        Config.model_name = model_var.get()
        Config.save_settings()
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
        description="Scan script files for malicious code using AI.",
        epilog="Examples:\n"
               "  python gptscan.py ./my_scripts --cli --use-gpt\n"
               "  python gptscan.py ./my_script.py --cli --json\n"
               "  python gptscan.py --git-changes --cli --fail-threshold 50",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {Config.VERSION}')
    parser.add_argument('target', nargs='?', help='The folder or file to scan.')
    parser.add_argument(
        'files',
        nargs='*',
        help='More folders or files to scan.'
    )

    scan_group = parser.add_argument_group("Scan Options")
    scan_group.add_argument('-p', '--path', type=str, help='An alternative way to specify the scan target.')
    scan_group.add_argument('-d', '--deep', action='store_true', help='Scan the whole file. Normally, it only checks the start and end to save time.')
    scan_group.add_argument('--dry-run', action='store_true', help='Show which files would be scanned without analyzing them.')
    scan_group.add_argument(
        '--extensions',
        type=str,
        help='Only scan these file types (comma-separated, e.g., "py,js").'
    )
    scan_group.add_argument(
        '-e', '--exclude',
        nargs='*',
        help='Skip files matching these patterns (e.g., "node_modules/*"). Patterns in .gptscanignore are also skipped.'
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
        '--fail-threshold',
        type=int,
        help='Exit with an error if any file has a threat score at or above this number (0-100).'
    )
    scan_group.add_argument(
        '--threshold', '-t',
        type=int,
        default=50,
        help='The lowest threat score to report (0-100). The default is 50.'
    )

    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument('-g', '--use-gpt', action='store_true', help='Use AI to create detailed reports for suspicious files. This requires an API key.')
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

    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--cli', action='store_true', help='Run in command-line mode instead of opening a window.')
    output_group.add_argument('-a', '--show-all', action='store_true', help='Show all scanned files, including safe ones.')
    output_group.add_argument('-o', '--output', type=str, help='Save the scan results to a file. The tool chooses the format based on the file extension.')
    output_group.add_argument('-j', '--json', action='store_true', help='Output results in JSON format (one object per line).')
    output_group.add_argument('--csv', action='store_true', help='Output results in CSV format (default).')
    output_group.add_argument('--sarif', action='store_true', help='Save results in SARIF format, a standard for security tools.')
    output_group.add_argument('--html', action='store_true', help='Create an HTML report of the results.')
    output_group.add_argument('--md', '--markdown', action='store_true', dest='markdown', help='Create a Markdown report of the results.')

    args = parser.parse_args()

    Config.provider = args.provider
    if args.api_base:
        Config.api_base = args.api_base

    if args.model:
        Config.model_name = args.model
    elif Config.provider == 'ollama':
        Config.model_name = 'llama3.2'

    if args.extensions:
        extension_list = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
        Config.set_extensions(extension_list, missing=False)

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
            git_files = get_git_changed_files()
            if not git_files:
                print("No git changes detected or not a git repository.", file=sys.stderr)
            scan_targets.extend(git_files)

        if not scan_targets:
            # Default to current directory if no targets provided
            scan_targets = ["."]

        output_format = 'csv'
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
            output_file=args.output
        )
        if args.fail_threshold is not None and threats > 0:
            sys.exit(1)
    else:
        app_root = create_gui(initial_path=scan_target)
        app_root.mainloop()

if __name__ == "__main__":
    main()
