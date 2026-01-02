import asyncio
import csv
import json
import os
import queue
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
    except FileNotFoundError:
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

    DEFAULT_EXTENSIONS = ['.py', '.js', '.bat', '.ps1']

    apikey_missing_message = (
        "API key not found. You can still scan locally, but AI analysis won't be available."
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
        cls.extensions_set = {ext.strip().lower() for ext in extensions_list if ext.strip()}

    @classmethod
    def initialize(cls) -> None:
        if not cls.apikey:
            print(cls.apikey_missing_message)
        if not cls.taskdesc:
            print(cls.task_missing_message)

        # Enable GPT if task description is present.
        # Specific provider requirements (like API key) are checked at runtime.
        cls.GPT_ENABLED = bool(cls.taskdesc)

        loaded_extensions = load_file('extensions.txt', mode='multi_line')
        if not loaded_extensions:
            cls.set_extensions(cls.DEFAULT_EXTENSIONS, missing=True)
            print(cls.extensions_missing_message)
        else:
            cls.set_extensions(loaded_extensions)


Config.initialize()

ui_queue = queue.Queue()
current_cancel_event: Optional[threading.Event] = None
_model_cache: Optional[Any] = None
_tf_module: Optional[Any] = None
_model_lock = threading.Lock()
_async_openai_client: Optional[Any] = None


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
            print(f"An unexpected error occurred: {err}")
            break

        if isinstance(extracted_data, dict):
            json_data = extracted_data
        else:
            print(extracted_data)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            messages.append({"role": "user", "content": f"I encountered an issue: {extracted_data}. Could you correct your response?"})
            retries += 1

    if isinstance(json_data, dict):
        Config.gpt_cache[cache_key] = json_data
        return json_data

    print("Failed to obtain a valid response from GPT after multiple retries.")
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
        col_widths = [tree.column(cid)['width'] for cid in tree['columns']]

        for iid in tree.get_children():
            values = tree.item(iid)['values']
            new_vals = [adjust_newlines(v, w) for v, w in zip(values, col_widths)]
            tree.item(iid, values=new_vals)


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
    """Insert a row into the treeview with wrapped text."""

    col_widths = [tree.column(cid)['width'] for cid in tree['columns']]
    wrapped_values = [adjust_newlines(v, w) for v, w in zip(values, col_widths)]
    tree.insert("", tk.END, values=wrapped_values)


def set_scanning_state(is_scanning: bool) -> None:
    """Enable or disable controls based on scanning state."""

    scan_button.config(state="disabled" if is_scanning else "normal")
    cancel_button.config(state="normal" if is_scanning else "disabled")


def finish_scan_state() -> None:
    """Reset scanning controls when a scan finishes or is cancelled."""

    global current_cancel_event
    current_cancel_event = None
    set_scanning_state(False)
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

    scan_path = textbox.get()
    if not scan_path:
        messagebox.showerror("Scan Error", "Please select a directory to scan.")
        return

    if not dry_var.get() and not os.path.exists('scripts.h5'):
        messagebox.showerror("Scan Error", "Model file scripts.h5 not found.")
        return

    current_cancel_event = threading.Event()
    set_scanning_state(True)
    update_status("Starting scan...")
    scan_args = (
        scan_path,
        deep_var.get(),
        all_var.get(),
        gpt_var.get(),
        current_cancel_event,
        Config.RATE_LIMIT_PER_MINUTE,
        dry_var.get(),
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
    Generator[Tuple[str, Tuple[Any, ...]], None, None]
        Tuples indicating events:
        - ('progress', (current: int, total: int, status: Optional[str]))
        - ('result', (path: str, own_conf: str, admin: str, user: str, gpt: str, snippet: str))
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
    yield ('progress', (progress_count, total_progress, "Scanning..."))

    gpt_requests: List[Dict[str, Any]] = []

    for index, file_path in enumerate(file_list):
        if cancel_event.is_set():
            break
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
                progress_count = index + 1
                yield ('progress', (progress_count, total_progress, None))
                continue

            print(file_path)
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
                progress_count = index + 1
                yield ('progress', (progress_count, total_progress, None))
                continue


            resultchecks: List[float] = []
            maxconf = 0.0
            max_window_bytes = b""
            error_message: Optional[str] = None

            try:
                with open(file_path, 'rb') as f:
                    for offset, window in iter_windows(f, file_size, deep_scan):
                        if cancel_event.is_set():
                            break
                        print("Scanning at:", offset if offset > 0 else 0)
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
        progress_count = index + 1
        yield ('progress', (progress_count, total_progress, None))

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
            progress_count += 1
            yield ('progress', (progress_count, total_progress, None))


def run_scan(
    scan_path: str,
    deep_scan: bool,
    show_all: bool,
    use_gpt: bool,
    cancel_event: threading.Event,
    rate_limit: int = Config.RATE_LIMIT_PER_MINUTE,
    dry_run: bool = False,
) -> None:
    """Consume scan events and forward them to the UI thread.

    Parameters
    ----------
    scan_path : str
        Directory to scan.
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
    last_total: Optional[int] = None
    try:
        for event_type, data in scan_files(
            scan_path,
            deep_scan,
            show_all,
            use_gpt,
            cancel_event,
            rate_limit=rate_limit,
            max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
            dry_run=dry_run,
        ):
            if cancel_event.is_set():
                break
            if event_type == 'progress':
                current, total, status = data

                if total != last_total:
                    enqueue_ui_update(configure_progress, total)
                    last_total = total
                enqueue_ui_update(update_progress, current)
                if status:
                    print(status)
                    enqueue_ui_update(update_status, status)
            elif event_type == 'result':
                enqueue_ui_update(insert_tree_row, data)
    finally:
        enqueue_ui_update(finish_scan_state)


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
        conf_str = r.get("gpt_conf", "") or r.get("own_conf", "")
        conf = parse_percent(conf_str)
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
        Format of the output ('csv', 'json', or 'sarif'). Defaults to 'csv'.
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

    sarif_buffer = []

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
            record = dict(zip(keys, data))
            if output_format == 'json':
                print(json.dumps(record))
            elif output_format == 'sarif':
                sarif_buffer.append(record)
            else:
                writer.writerow(data)
        elif event_type == 'progress':
            current, total, status = data
            final_progress = (current, total)
            print(f"Scanning: {current}/{total} files\r", end='', file=sys.stderr)
            if status:
                print(f"{status}\r", end='', file=sys.stderr)

    if final_progress is not None:
        print(file=sys.stderr)

    if output_format == 'sarif':
        sarif_log = generate_sarif(sarif_buffer)
        print(json.dumps(sarif_log, indent=2))


def export_results() -> None:
    """Save the current Treeview contents to a CSV chosen by the user.

    Returns
    -------
    None
        Writes the Treeview rows to the selected CSV path or shows an error.
    """
    file_path = tkinter.filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Export Scan Results",
    )
    if not file_path:
        return

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(tree["columns"])
            for item_id in tree.get_children():
                writer.writerow(tree.item(item_id)["values"])
    except OSError as err:
        messagebox.showerror("Export Failed", f"Could not save results:\n{err}")


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
    global root, textbox, progress_bar, status_label, deep_var, all_var, gpt_var, dry_var, tree, scan_button, cancel_button

    root = tk.Tk()
    root.geometry("1000x600")
    root.title("GPT Virus Scanner")

    # Configure grid weights to ensure resizing behaves correctly
    root.columnconfigure(0, weight=1)
    root.rowconfigure(6, weight=1)  # The row containing the Treeview

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
    select_dir_btn = ttk.Button(input_frame, text="Select Directory", command=browse_button_click)
    select_dir_btn.grid(row=0, column=2, sticky="e", padx=(5, 0))

    # --- Options Frame ---
    options_frame = ttk.Frame(root)
    options_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

    deep_var = tk.BooleanVar()
    deep_checkbox = ttk.Checkbutton(options_frame, text="Deep scan", variable=deep_var)
    deep_checkbox.pack(side=tk.LEFT, padx=10)

    all_var = tk.BooleanVar()
    all_checkbox = ttk.Checkbutton(options_frame, text="Show all files", variable=all_var)
    all_checkbox.pack(side=tk.LEFT, padx=10)

    gpt_var = tk.BooleanVar()

    dry_var = tk.BooleanVar()
    dry_checkbox = ttk.Checkbutton(options_frame, text="Dry Run", variable=dry_var)
    dry_checkbox.pack(side=tk.LEFT, padx=10)

    # --- Provider Frame ---
    provider_frame = ttk.LabelFrame(root, text="AI Analysis")
    provider_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

    def toggle_ai_controls():
        enabled = gpt_var.get()
        if enabled:
            provider_combo.config(state="readonly")
            model_combo.config(state="normal")
        else:
            provider_combo.config(state="disabled")
            model_combo.config(state="disabled")

    gpt_checkbox = ttk.Checkbutton(provider_frame, text="Use AI Analysis", variable=gpt_var, command=toggle_ai_controls)
    gpt_checkbox.pack(side=tk.LEFT, padx=10)

    if not Config.GPT_ENABLED:
        gpt_var.set(False)
        gpt_checkbox.config(state="disabled")
        messagebox.showwarning("GPT Disabled",
                                       "task.txt not found. GPT functionality is disabled.")

    ttk.Label(provider_frame, text="Provider:").pack(side=tk.LEFT, padx=5, pady=5)
    provider_var = tk.StringVar(value=Config.provider)
    provider_combo = ttk.Combobox(provider_frame, textvariable=provider_var, values=["openai", "openrouter", "ollama"], state="readonly", width=12)
    provider_combo.pack(side=tk.LEFT, padx=5, pady=5)

    ttk.Label(provider_frame, text="Model:").pack(side=tk.LEFT, padx=5, pady=5)
    model_var = tk.StringVar(value=Config.model_name)
    model_combo = ttk.Combobox(provider_frame, textvariable=model_var, width=20)
    model_combo.pack(side=tk.LEFT, padx=5, pady=5)

    toggle_ai_controls()

    def update_model_presets(provider: str):
        if provider == "openai":
            model_combo['values'] = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif provider == "ollama":
            model_combo['values'] = ["llama3.2", "llama3.1", "deepseek-r1", "phi4", "mistral", "gemma2"]
        elif provider == "openrouter":
            model_combo['values'] = ["gpt-4o", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-r1", "google/gemini-flash-1.5", "meta-llama/llama-3.3-70b-instruct"]
        else:
            model_combo['values'] = []

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
    action_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

    scan_button = ttk.Button(action_frame, text="Scan now", command=button_click)
    scan_button.pack(side=tk.LEFT, padx=5)
    cancel_button = ttk.Button(action_frame, text="Cancel", command=cancel_scan, state="disabled")
    cancel_button.pack(side=tk.LEFT, padx=5)
    export_button = ttk.Button(action_frame, text="Export CSV", command=export_results)
    export_button.pack(side=tk.RIGHT, padx=5)

    # --- Progress Bar ---
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, mode='determinate')
    progress_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=5)

    status_label = ttk.Label(root, text="Ready", anchor="w")
    status_label.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 5))

    # --- Treeview ---
    style = ttk.Style(root)
    style.configure('Scanner.Treeview', rowheight=50)

    tree_frame = ttk.Frame(root)
    tree_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=5)
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    tree = ttk.Treeview(tree_frame, style='Scanner.Treeview')
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
    tree.grid(row=0, column=0, sticky="nsew")

    motion_handler(tree, None)   # Perform initial wrapping
    set_scanning_state(False)
    return root


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT Virus Scanner")
    parser.add_argument('target', nargs='?', help='The file or folder to check.')
    parser.add_argument(
        'files',
        nargs='*',
        help='Additional files to check.'
    )

    scan_group = parser.add_argument_group("Scan Configuration")
    scan_group.add_argument('--path', type=str, help='The folder to scan.')
    scan_group.add_argument('--deep', action='store_true', help='Check the whole file, not just the start and end (slower).')
    scan_group.add_argument('--dry-run', action='store_true', help='List the files that would be checked, without actually scanning them.')
    scan_group.add_argument(
        '--extensions',
        type=str,
        help='Only check files ending with these extensions (e.g., .py, .js).'
    )
    scan_group.add_argument(
        '--exclude',
        nargs='*',
        help='Patterns to exclude from scan (e.g., node_modules/*, *.test.py).'
    )
    scan_group.add_argument(
        '--file-list',
        type=argparse.FileType('r'),
        help='Read list of files to scan from a file (use "-" for stdin).'
    )

    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument('--use-gpt', action='store_true', help='Ask the AI to explain suspicious code.')
    ai_group.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'openrouter', 'ollama'],
        help='Choose the AI provider (default: openai).'
    )
    ai_group.add_argument(
        '--model',
        type=str,
        help='The specific AI model to use (e.g., gpt-4o, llama3.2).'
    )
    ai_group.add_argument(
        '--api-base',
        type=str,
        help='Custom URL for the API server.'
    )
    ai_group.add_argument(
        '--rate-limit',
        type=int,
        default=Config.RATE_LIMIT_PER_MINUTE,
        help='Max AI requests per minute (default: 60).'
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument('--cli', action='store_true', help='Run without the graphical window.')
    output_group.add_argument('--show-all', action='store_true', help='List every file, even if it looks safe.')
    output_group.add_argument('--json', action='store_true', help='Print results in JSON format instead of CSV.')
    output_group.add_argument('--sarif', action='store_true', help='Output results in SARIF format (used by other security tools).')

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

        if not scan_targets:
            parser.error('You must provide a file or folder to scan.')

        output_format = 'csv'
        if args.json:
            output_format = 'json'
        if args.sarif:
            output_format = 'sarif'
        run_cli(scan_targets, args.deep, args.show_all, args.use_gpt, args.rate_limit, output_format=output_format, dry_run=args.dry_run, exclude_patterns=args.exclude)
    else:
        app_root = create_gui(initial_path=scan_target)
        app_root.mainloop()

if __name__ == "__main__":
    main()
