import asyncio
import csv
import html
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
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
tree: Optional[ttk.Treeview] = None
scan_button: Optional[ttk.Button] = None
cancel_button: Optional[ttk.Button] = None
context_menu: Optional[tk.Menu] = None


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

    DEFAULT_EXTENSIONS = ['.py', '.js', '.bat', '.ps1']

    apikey_missing_message = (
        "Note: No API key found. AI Analysis (OpenAI/OpenRouter) requires one, but local AI (Ollama) works without it."
    )
    task_missing_message = (
        "Task file missing. AI analysis will be skipped."
    )
    extensions_missing_message = (
        f"Extensions file missing. Using defaults: {', '.join(DEFAULT_EXTENSIONS)}"
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
    def initialize(cls) -> None:
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
    status_label.config(text=message)
    root.update_idletasks()


def update_progress(value: int) -> None:
    """Update the progress bar to reflect current progress.

    Parameters
    ----------
    value : int
        Current progress count to display.
    """
    progress_bar['value'] = value
    root.update_idletasks()


def configure_progress(max_value: int) -> None:
    """Initialize progress bar values for a new scan.

    Parameters
    ----------
    max_value : int
        Total number of steps expected for the scan.
    """
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
    while not ui_queue.empty():
        func, args, kwargs = ui_queue.get()
        try:
            func(*args, **kwargs)
        finally:
            ui_queue.task_done()
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


def browse_button_click() -> None:
    """Handle the directory selection dialog and populate the textbox.

    Returns
    -------
    None
        The selected folder path is written into the GUI textbox.
    """
    folder_selected = tkinter.filedialog.askdirectory()
    if folder_selected:
        textbox.delete(0, tk.END)
        textbox.insert(0, folder_selected)


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
    expected_keys = set(Config.EXPECTED_KEYS)

    content = response.choices[0].message.content

    try:
        json_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return str(exc)

    if not isinstance(json_data, dict):
        return "Response JSON must be an object with expected keys."

    missing_keys = [key for key in Config.EXPECTED_KEYS if key not in json_data]
    if missing_keys:
        return f"Missing keys: {', '.join(missing_keys)}"

    extra_keys = set(json_data.keys()) - expected_keys
    if extra_keys:
        return f"Unexpected keys present: {', '.join(sorted(extra_keys))}"

    threat_level_value = json_data.get("threat-level")
    try:
        threat_level = int(threat_level_value)
    except (TypeError, ValueError):
        return f"The 'threat-level' value '{threat_level_value}' is not a valid integer."

    if not 0 <= threat_level <= 100:
        return f"The 'threat-level' value {threat_level} is not between 0 and 100 inclusive."

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


def motion_handler(tree: ttk.Treeview, event: Optional[tk.Event]) -> None:
    """Wrap long cell values so they fit within the visible column width."""

    if (event is None) or (tree.identify_region(event.x, event.y) == "separator"):
        measure = default_font_measure or tkinter.font.Font(font='TkDefaultFont').measure
        col_widths = [tree.column(cid)['width'] for cid in tree['columns']]

        for iid in tree.get_children():
            values = tree.item(iid)['values']
            new_vals = [adjust_newlines(v, w, measure=measure) for v, w in zip(values, col_widths)]
            tree.item(iid, values=new_vals)


def get_git_changed_files(path: str = ".") -> List[str]:
    """Get a list of changed files (staged, unstaged, untracked) from git."""
    files = set()
    try:
        # Changed (staged and unstaged) relative to HEAD
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=path,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())

        # Untracked
        output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=path,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        files.update(line.strip() for line in output.splitlines() if line.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not found or not a repo, or error running git
        # We assume the caller handles logic (e.g. "No git changes detected") based on empty result,
        # or warns if it was explicitly requested but failed.
        pass

    return [os.path.join(path, f) for f in files if os.path.exists(os.path.join(path, f))]


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


def format_scan_summary(total_scanned: int, threats_found: int, total_bytes: Optional[int] = None, elapsed_time: Optional[float] = None) -> str:
    """Format a human-readable summary of the scan results."""
    threat_text = "suspicious file" if threats_found == 1 else "suspicious files"
    summary = f"Scan complete: {total_scanned} files scanned, {threats_found} {threat_text} found."

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


def insert_tree_row(values: Tuple[Any, ...]) -> None:
    """Insert a row into the treeview with wrapped text and highlighting."""
    measure = default_font_measure or tkinter.font.Font(font='TkDefaultFont').measure
    col_widths = [tree.column(cid)['width'] for cid in tree['columns']]
    wrapped_values = [adjust_newlines(v, w, measure=measure) for v, w in zip(values, col_widths)]

    # Determine risk level based on confidence scores
    # data format: (path, own_conf, admin, user, gpt_conf, snippet)
    conf = get_effective_confidence(values[1], values[4])

    tag = ''
    if conf > 80:
        tag = 'high-risk'
    elif conf > 50:
        tag = 'medium-risk'

    tree.insert("", tk.END, values=wrapped_values, tags=(tag,) if tag else ())


def set_scanning_state(is_scanning: bool) -> None:
    """Enable or disable controls based on scanning state."""

    scan_button.config(state="disabled" if is_scanning else "normal")
    cancel_button.config(state="normal" if is_scanning else "disabled")


def finish_scan_state(total_scanned: Optional[int] = None, threats_found: Optional[int] = None, total_bytes: Optional[int] = None, elapsed_time: Optional[float] = None) -> None:
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
    """

    global current_cancel_event
    current_cancel_event = None
    set_scanning_state(False)

    if total_scanned is not None and threats_found is not None:
        summary = format_scan_summary(total_scanned, threats_found, total_bytes, elapsed_time)
        update_status(summary)
    else:
        update_status("Ready")


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
    if tree:
        tree.delete(*tree.get_children())

    scan_path = textbox.get()
    if not scan_path:
        messagebox.showerror("Scan Error", "Please select a directory to scan.")
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

        extension = file_path.suffix.lower()
        if extension in Config.extensions_set:
            if dry_run:
                yield (
                    'result',
                    (
                        str(file_path),
                        'Dry Run',
                        '',
                        '',
                        '',
                        '(File would be scanned)',
                    )
                )
            else:
                print(file_path, file=sys.stderr)
                try:
                    file_size = file_path.stat().st_size
                except OSError as err:
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
                    file_size = None

                if file_size is not None:
                    total_bytes_scanned += file_size
                    resultchecks: List[float] = []
                    maxconf = 0.0
                    max_window_bytes = b""
                    error_message: Optional[str] = None

                    try:
                        with open(file_path, 'rb') as f:
                            for offset, window in iter_windows(f, file_size, deep_scan):
                                if cancel_event.is_set():
                                    break
                                print("Scanning at:", offset if offset > 0 else 0, file=sys.stderr)
                                result, padded_bytes = predict_window(window)
                                resultchecks.append(result)
                                if result > maxconf:
                                    maxconf = result
                                    max_window_bytes = padded_bytes
                    except OSError as err:
                        error_message = f"Error reading file: {err}"

                    best_result = max(resultchecks, default=0)
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
                    elif resultchecks:
                        percent = f"{maxconf:.0%}"
                        snippet = ''.join(map(chr, max_window_bytes)).strip()
                        cleaned_snippet = ''.join([s for s in snippet.strip().splitlines(True) if s.strip()])
                        if best_result > .5 and use_gpt and Config.GPT_ENABLED:
                            gpt_requests.append(
                                {
                                    "path": str(file_path),
                                    "percent": percent,
                                    "snippet": snippet,
                                    "cleaned_snippet": cleaned_snippet,
                                }
                            )
                        elif best_result > .5 or show_all:
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

                status_text = f"{status} ({current}/{total})" if status else f"Scanning: {current}/{total}"
                print(status_text, file=sys.stderr)
                enqueue_ui_update(update_status, status_text)
            elif event_type == 'result':
                # data format: (path, own_conf, admin, user, gpt_conf, snippet)
                conf = get_effective_confidence(data[1], data[4])

                if conf > 50:
                    threats_found += 1

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
            metrics.get('elapsed_time')
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
        elif conf > 50:
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
        elif conf_val > 50:
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


def run_cli(targets: Union[str, List[str]], deep: bool, show_all: bool, use_gpt: bool, rate_limit: int, output_format: str = 'csv', dry_run: bool = False, exclude_patterns: Optional[List[str]] = None) -> None:
    """Run scans and stream results to stdout.

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
        Format of the output ('csv', 'json', 'sarif', or 'html'). Defaults to 'csv'.
    dry_run : bool
        Whether to simulate the scan.
    exclude_patterns : List[str], optional
        List of glob patterns to exclude from the scan.
    """
    keys = ["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"]

    if output_format == 'csv':
        writer = csv.writer(sys.stdout)
        writer.writerow(keys)

    cancel_event = threading.Event()
    final_progress: Optional[Tuple[int, int]] = None
    threats_found = 0
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
            if conf > 50:
                threats_found += 1

            record = dict(zip(keys, data))
            if output_format == 'json':
                print(json.dumps(record))
            elif output_format in ('sarif', 'html'):
                result_buffer.append(record)
            else:
                writer.writerow(data)
        elif event_type == 'progress':
            current, total, status = data
            final_progress = (current, total)
            print(f"Scanning: {current}/{total} files\r", end='', file=sys.stderr)
            if status:
                print(f"{status}\r", end='', file=sys.stderr)
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
            metrics.get('elapsed_time')
        )
        print(summary, file=sys.stderr)

    if output_format == 'sarif':
        sarif_log = generate_sarif(result_buffer)
        print(json.dumps(sarif_log, indent=2))
    elif output_format == 'html':
        print(generate_html(result_buffer))


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
            ("All supported formats", "*.json;*.jsonl;*.ndjson;*.csv"),
            ("JSON files", "*.json;*.jsonl;*.ndjson"),
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

        if ext in ('.json', '.jsonl', '.ndjson'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("File is empty.")

                if content.startswith('['):
                    # Standard JSON list
                    data_to_import = json.loads(content)
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
        tree.delete(*tree.get_children())

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

        update_status(f"Imported {count} results from {os.path.basename(file_path)}")

    except Exception as err:
        messagebox.showerror("Import Failed", f"Could not load results:\n{err}")


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
            ("HTML files", "*.html"),
            ("JSON files", "*.json"),
            ("SARIF files", "*.sarif"),
            ("All files", "*.*")
        ],
        title="Export Scan Results",
    )
    if not file_path:
        return

    # Collect data from Treeview and unwrap newlines added for display
    columns = tree["columns"]
    results = []
    for item_id in tree.get_children():
        values = list(tree.item(item_id)["values"])
        results.append(dict(zip(columns, values))) # Use original values for JSON/HTML/SARIF as they handle newlines better

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.json':
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif ext == '.html':
            # Need to map keys for generate_html
            mapped_results = []
            for r in results:
                mapped_results.append({
                    "path": r["path"],
                    "own_conf": r["own_conf"],
                    "admin_desc": r["admin_desc"],
                    "end-user_desc": r["end-user_desc"],
                    "gpt_conf": r["gpt_conf"],
                    "snippet": r["snippet"]
                })
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(generate_html(mapped_results))
        elif ext == '.sarif':
            sarif_log = generate_sarif(results)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sarif_log, f, indent=2)
        else: # Default to CSV
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(columns)
                for item_id in tree.get_children():
                    # For CSV, we definitely want to unwrap or it looks messy in Excel
                    vals = [str(v).replace('\n', ' ') for v in tree.item(item_id)["values"]]
                    writer.writerow(vals)

        messagebox.showinfo("Export Successful", f"Results saved to {os.path.basename(file_path)}")

    except Exception as err:
        messagebox.showerror("Export Failed", f"Could not save results:\n{err}")


def open_file(event: Optional[tk.Event] = None) -> None:
    """Open the selected file in the system's default application."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return

    # Get the file path from the first column of the selected row
    item_id = selection[0]
    values = tree.item(item_id, "values")
    if not values:
        return

    file_path = str(values[0]).replace('\n', '') # Remove wrapping
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
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return
    item_id = selection[0]
    values = tree.item(item_id, "values")
    if not values:
        return
    file_path = str(values[0]).replace('\n', '') # Remove display wrapping
    tree.clipboard_clear()
    tree.clipboard_append(file_path)


def copy_snippet() -> None:
    """Copy the selected row's code snippet to the clipboard."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return
    item_id = selection[0]
    values = tree.item(item_id, "values")
    if not values:
        return
    # Snippet is the last column
    snippet = str(values[-1])
    tree.clipboard_clear()
    tree.clipboard_append(snippet)


def show_in_folder() -> None:
    """Reveal the selected file in the system file manager."""
    if not tree:
        return
    selection = tree.selection()
    if not selection:
        return
    item_id = selection[0]
    values = tree.item(item_id, "values")
    if not values:
        return
    file_path = str(values[0]).replace('\n', '')
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
        if iid:
            tree.selection_set(iid)

    if tree.selection():
        context_menu.post(event.x_root, event.y_root)


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
    global root, textbox, progress_bar, status_label, deep_var, all_var, gpt_var, dry_var, git_var, tree, scan_button, cancel_button, default_font_measure

    root = tk.Tk()
    root.geometry("1000x600")
    root.title("GPT Virus Scanner")
    default_font_measure = tkinter.font.Font(font='TkDefaultFont').measure

    # Configure grid weights to ensure resizing behaves correctly
    root.columnconfigure(0, weight=1)
    root.rowconfigure(5, weight=1)  # The row containing the Treeview

    # --- Input Frame ---
    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    input_frame.columnconfigure(1, weight=1)

    ttk.Label(input_frame, text="Path to scan:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    textbox = ttk.Entry(input_frame)
    path_to_use = initial_path if initial_path else os.getcwd()
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
    select_dir_btn = ttk.Button(input_frame, text="Select Directory...", command=browse_button_click)
    select_dir_btn.grid(row=0, column=2, sticky="e", padx=(5, 0))
    bind_hover_message(select_dir_btn, "Browse for a directory to scan.")

    # --- Settings Container ---
    settings_frame = ttk.Frame(root)
    settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

    # --- Options Frame ---
    options_frame = ttk.LabelFrame(settings_frame, text="Scan Options")
    options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

    gpt_var = tk.BooleanVar()

    git_var = tk.BooleanVar()
    git_checkbox = ttk.Checkbutton(options_frame, text="Git changes only", variable=git_var)
    git_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(git_checkbox, "Only scan files that have been modified or are untracked in Git.")

    deep_var = tk.BooleanVar()
    deep_checkbox = ttk.Checkbutton(options_frame, text="Deep scan", variable=deep_var)
    deep_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(deep_checkbox, "Scan the entire file content (slower). Default scans only start/end.")

    all_var = tk.BooleanVar()
    all_checkbox = ttk.Checkbutton(options_frame, text="Show all files", variable=all_var)
    all_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(all_checkbox, "Display all scanned files, including safe ones.")

    dry_var = tk.BooleanVar()
    dry_checkbox = ttk.Checkbutton(options_frame, text="Dry Run", variable=dry_var)
    dry_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(dry_checkbox, "Simulate the scan process without running checks.")

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

    gpt_checkbox = ttk.Checkbutton(provider_frame, text="Use AI Analysis", variable=gpt_var, command=toggle_ai_controls)
    gpt_checkbox.pack(side=tk.TOP, anchor='w', padx=10, pady=2)
    bind_hover_message(gpt_checkbox, "Send suspicious code to AI for explanation.")

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

    # --- Action Frame ---
    action_frame = ttk.Frame(root)
    action_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

    scan_button = ttk.Button(action_frame, text="Scan now", command=button_click, default='active')
    scan_button.pack(side=tk.LEFT, padx=5)
    bind_hover_message(scan_button, "Start the scan.")

    cancel_button = ttk.Button(action_frame, text="Cancel", command=cancel_scan, state="disabled")
    cancel_button.pack(side=tk.LEFT, padx=5)
    bind_hover_message(cancel_button, "Stop the current scan.")

    export_button = ttk.Button(action_frame, text="Export Results...", command=export_results)
    export_button.pack(side=tk.RIGHT, padx=5)
    bind_hover_message(export_button, "Save results to CSV, HTML, JSON, or SARIF.")

    import_button = ttk.Button(action_frame, text="Import Results...", command=import_results)
    import_button.pack(side=tk.RIGHT, padx=5)
    bind_hover_message(import_button, "Load results from a JSON or CSV file.")

    # --- Progress Bar ---
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, mode='determinate')
    progress_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

    status_label = ttk.Label(root, text="Ready", anchor="w")
    status_label.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 5))

    # --- Treeview ---
    style = ttk.Style(root)
    style.configure('Scanner.Treeview', rowheight=50)

    # Configure tags for row highlighting
    # Note: 'alt' theme or similar might be needed for background colors to show in some environments
    tree_frame = ttk.Frame(root)
    tree_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=5)
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    tree = ttk.Treeview(tree_frame, style='Scanner.Treeview')
    tree.tag_configure('high-risk', background='#ffcccc')
    tree.tag_configure('medium-risk', background='#fff0cc')
    tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("path", width=150, stretch=tk.YES, anchor="w")
    tree.column("own_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("admin_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("end-user_desc", width=150, stretch=tk.YES, anchor="w")
    tree.column("gpt_conf", width=80, stretch=tk.NO, anchor="e")
    tree.column("snippet", width=150, stretch=tk.YES, anchor="w")
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
    tree.bind('<Double-1>', open_file)
    tree.bind('<Return>', open_file)
    tree.grid(row=0, column=0, sticky="nsew")

    # --- Context Menu ---
    global context_menu
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="Open File", command=open_file)
    context_menu.add_command(label="Show in Folder", command=show_in_folder)
    context_menu.add_separator()
    context_menu.add_command(label="Copy File Path", command=copy_path)
    context_menu.add_command(label="Copy Snippet", command=copy_snippet)

    # Bind context menu to right-click and menu key
    tree.bind('<Button-3>', show_context_menu) # Windows/Linux
    tree.bind('<Button-2>', show_context_menu) # macOS
    tree.bind('<Menu>', show_context_menu)

    motion_handler(tree, None)   # Perform initial wrapping
    set_scanning_state(False)
    return root


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT Virus Scanner")
    parser.add_argument('target', nargs='?', help='The file or folder to scan.')
    parser.add_argument(
        'files',
        nargs='*',
        help='Additional files or folders to scan.'
    )

    scan_group = parser.add_argument_group("Scan Configuration")
    scan_group.add_argument('--path', type=str, help='Alternative way to specify the folder to scan.')
    scan_group.add_argument('--deep', action='store_true', help='Scan the entire file content instead of just the first and last parts. This is slower but more thorough.')
    scan_group.add_argument('--dry-run', action='store_true', help='Simulate the scan to see which files would be checked, without running the AI models.')
    scan_group.add_argument(
        '--extensions',
        type=str,
        help='Only scan files with these specific extensions (e.g., .py, .js).'
    )
    scan_group.add_argument(
        '--exclude',
        nargs='*',
        help='Skip files that match these patterns (e.g., node_modules/*, *.test.py). Files in .gptscanignore are also skipped.'
    )
    scan_group.add_argument(
        '--file-list',
        type=argparse.FileType('r'),
        help='Read a list of files to scan from a text file (use "-" to read from standard input).'
    )
    scan_group.add_argument(
        '--git-changes',
        action='store_true',
        help='Only scan files that have been modified or are untracked in the current git repository.'
    )

    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument('--use-gpt', action='store_true', help='Send suspicious code to the AI provider for a detailed explanation.')
    ai_group.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'openrouter', 'ollama'],
        help='Select the AI provider to use for analysis (default: openai).'
    )
    ai_group.add_argument(
        '--model',
        type=str,
        help='Specify the exact AI model to use (e.g., gpt-4o, llama3.2).'
    )
    ai_group.add_argument(
        '--api-base',
        type=str,
        help='Use a custom URL for the API server (useful for proxies or local servers).'
    )
    ai_group.add_argument(
        '--rate-limit',
        type=int,
        default=Config.RATE_LIMIT_PER_MINUTE,
        help='Limit the number of AI requests per minute to avoid errors (default: 60).'
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument('--cli', action='store_true', help='Run without the graphical window.')
    output_group.add_argument('--show-all', action='store_true', help='List every file, even if it looks safe.')
    output_group.add_argument('--json', action='store_true', help='Output results in JSON Lines (NDJSON) format instead of CSV.')
    output_group.add_argument('--sarif', action='store_true', help='Output results in SARIF format (standard for security tools).')
    output_group.add_argument('--html', action='store_true', help='Output results as a standalone HTML report.')

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
            parser.error('You must provide a file or folder to scan.')

        output_format = 'csv'
        if args.json:
            output_format = 'json'
        elif args.sarif:
            output_format = 'sarif'
        elif args.html:
            output_format = 'html'

        final_excludes = list(set((Config.ignore_patterns or []) + (args.exclude or [])))
        run_cli(scan_targets, args.deep, args.show_all, args.use_gpt, args.rate_limit, output_format=output_format, dry_run=args.dry_run, exclude_patterns=final_excludes)
    else:
        app_root = create_gui(initial_path=scan_target)
        app_root.mainloop()

if __name__ == "__main__":
    main()
