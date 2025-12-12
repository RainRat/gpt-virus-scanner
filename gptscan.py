import csv
import json
import queue
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import tkinter as tk
import tkinter.filedialog
import tkinter.font
from tkinter import messagebox
import tkinter.ttk as ttk


def load_file(filename: str, mode: str = 'single_line') -> Union[str, List[str]]:
    """Read content from a file in either single-line or multi-line mode.

    Parameters
    ----------
    filename : str
        Path to the file to read.
    mode : str, optional
        Reading mode; ``"single_line"`` returns the first line, while
        ``"multi_line"`` returns all lines as a list. Defaults to ``"single_line"``.

    Returns
    -------
    Union[str, List[str]]
        The requested file content, or an empty string if the file is missing.
    """
    try:
        with open(filename, 'r') as file:
            if mode == 'single_line':
                return file.readline().strip()
            elif mode == 'multi_line':
                return file.read().splitlines()
    except FileNotFoundError:
        return ''


class Config:
    MAXLEN = 1024
    EXPECTED_KEYS = ["administrator", "end-user", "threat-level"]
    MAX_RETRIES = 3
    gpt_cache: Dict[int, Dict[str, Any]] = {}
    apikey: str = load_file('apikey.txt')
    taskdesc: str = load_file('task.txt')
    GPT_ENABLED: bool = False
    extensions: Union[List[str], str] = []
    extensions_set: set[str] = set()
    extensions_missing: bool = False

    apikey_missing_message = (
        "OpenAI key file not found. No GPT data will be included in report..."
    )
    task_missing_message = (
        "Task description file not found. No GPT data will be included in report..."
    )
    extensions_missing_message = (
        "Extensions list not found! Using default extensions: .py, .js, .bat, .ps1"
    )

    @classmethod
    def set_extensions(cls, extensions_list: List[str], missing: bool = False) -> None:
        cls.extensions_missing = missing
        cls.extensions = extensions_list
        cls.extensions_set = {ext.strip().lower() for ext in extensions_list if ext.strip()}

    @classmethod
    def initialize(cls) -> None:
        if not cls.apikey:
            print(cls.apikey_missing_message)
        if not cls.taskdesc:
            print(cls.task_missing_message)
            cls.apikey = ''
        cls.GPT_ENABLED = bool(cls.apikey and cls.taskdesc)

        loaded_extensions = load_file('extensions.txt', mode='multi_line')
        if not loaded_extensions:
            cls.set_extensions(['.py', '.js', '.bat', '.ps1'], missing=True)
            print(cls.extensions_missing_message)
        else:
            cls.set_extensions(loaded_extensions)


Config.initialize()

ui_queue = queue.Queue()
current_cancel_event: Optional[threading.Event] = None

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
    textbox.delete(0, tk.END)
    textbox.insert(0, folder_selected)


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
    try:
        json_data = json.loads(response.choices[0].message.content)
        missing_keys = [key for key in Config.EXPECTED_KEYS if key not in json_data]
        if missing_keys:
            raise ValueError(f"Missing keys: {', '.join(missing_keys)}")

        threat_level_str = json_data.get("threat-level", None)
        try:
            threat_level = int(threat_level_str)
        except ValueError:
            raise ValueError(f"The 'threat-level' value '{threat_level_str}' is not a valid integer.")

        if not 0 <= threat_level <= 100:
            raise ValueError(f"The 'threat-level' value {threat_level} is not between 0 and 100 inclusive.")
        return json_data
    except (json.JSONDecodeError, ValueError) as e:
        return str(e)


def handle_gpt_response(snippet: str, taskdesc: str) -> Optional[Dict]:
    """Request GPT analysis for a snippet with retry and caching support.

    Parameters
    ----------
    snippet : str
        The code or text snippet to analyze.
    taskdesc : str
        Prompt content that instructs GPT on how to respond.

    Returns
    -------
    Optional[Dict]
        Parsed JSON data containing administrator/end-user descriptions and
        threat level, or ``None`` if retries are exhausted.
    """
    from openai import OpenAI
    retries = 0
    json_data = None
    client = OpenAI(api_key=Config.apikey)
    create_completion = partial(client.chat.completions.create, model="gpt-3.5-turbo")
    cache_key = hash(snippet)
    if cache_key in Config.gpt_cache:
        return Config.gpt_cache[cache_key]
    messages = [
        {"role": "system", "content": taskdesc},
        {"role": "user", "content": snippet}
    ]
    while retries < Config.MAX_RETRIES and (json_data is None or isinstance(json_data, str)):

        try:
            response = create_completion(messages=messages)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

        if response:
            extracted_data = extract_data_from_gpt_response(response)
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
    else:
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
        if measure(' '.join(line)) < (width - pad):
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


def list_files(path: str) -> List[Path]:
    """Recursively collect files under a directory path.

    Parameters
    ----------
    path : str
        Root directory to traverse.

    Returns
    -------
    List[Path]
        All files found under ``path``.
    """
    path = Path(path)
    return [p for p in path.rglob('*') if p.is_file()]


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
    else:
        values_with_ids = [(val, k) for val, k in values_with_ids]

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

    current_cancel_event = threading.Event()
    set_scanning_state(True)
    scan_args = (
        scan_path,
        deep_var.get(),
        all_var.get(),
        gpt_var.get(),
        current_cancel_event,
    )
    scan_thread = threading.Thread(target=run_scan, args=scan_args, daemon=True)
    scan_thread.start()


def cancel_scan() -> None:
    """Signal the active scan to stop."""

    if current_cancel_event:
        current_cancel_event.set()


def scan_files(
    scan_path: str,
    deep_scan: bool,
    show_all: bool,
    use_gpt: bool,
    cancel_event: Optional[threading.Event] = None,
) -> Generator[Tuple[str, Tuple[Any, ...]], None, None]:
    """Scan files for malicious content and optionally request GPT analysis.

    Parameters
    ----------
    scan_path : str
        Directory path to search for files.
    deep_scan : bool
        Whether to scan overlapping 1024-byte windows beyond the first block.
    show_all : bool
        Whether to yield all scanned files regardless of confidence threshold.
    use_gpt : bool
        Whether to request GPT analysis when the local model is confident.

    Yields
    ------
    Generator[Tuple[str, Tuple[Any, ...]], None, None]
        Tuples indicating either progress updates or result rows for the UI/CLI.
    """
    import tensorflow as tf
    cancel_event = cancel_event or threading.Event()
    modelscript = tf.keras.models.load_model('scripts.h5', compile=False)
    file_list = list_files(scan_path)
    yield ('progress', 0, len(file_list))

    for index, file_path in enumerate(file_list):
        if cancel_event.is_set():
            break
        extension = Path(file_path).suffix.lower()
        if extension in Config.extensions_set:
            print(file_path)
            with open(file_path, 'rb') as f:
                data = list(f.read())
            resultchecks = []
            if len(data) <= Config.MAXLEN:
                numtoadd = Config.MAXLEN - len(data)
                data.extend([13] * numtoadd)
            file_size = max(Config.MAXLEN, len(data))
            tf_data = tf.expand_dims(tf.constant(data), axis=0)

            maxconf_pos = 0
            maxconf = 0
            for i in range(0, file_size - Config.MAXLEN + 1, Config.MAXLEN):
                if cancel_event.is_set():
                    break
                if i >= Config.MAXLEN and not deep_scan:
                    continue
                print("Scanning at:", i)
                result = modelscript.predict(tf_data[:, i:i + 1024], batch_size=1, steps=1)[0][0]
                resultchecks.append(result)
                if result > maxconf:
                    maxconf_pos = i
                    maxconf = result

            if file_size > Config.MAXLEN and not cancel_event.is_set():
                print("Scanning at:", -Config.MAXLEN)
                result = modelscript.predict(tf_data[:, -Config.MAXLEN:], batch_size=1, steps=1)[0][0]
                resultchecks.append(result)
                if result > maxconf:
                    maxconf_pos = file_size - Config.MAXLEN
                    maxconf = result

            percent = f"{maxconf:.0%}"
            snippet = ''.join(map(chr, bytes(data[maxconf_pos:maxconf_pos + 1024]))).strip()
            if max(resultchecks) > .5 and use_gpt and Config.GPT_ENABLED:
                json_data = handle_gpt_response(snippet, Config.taskdesc)
                if json_data is None:
                    admin_desc = 'JSON Parse Error'
                    enduser_desc = 'JSON Parse Error'
                    chatgpt_conf_percent = 'JSON Parse Error'
                else:
                    admin_desc = json_data["administrator"]
                    enduser_desc = json_data["end-user"]
                    chatgpt_conf_percent = "{:.0%}".format(int(json_data["threat-level"]) / 100.)
            else:
                admin_desc = ''
                enduser_desc = ''
                chatgpt_conf_percent = ''
            snippet = ''.join([s for s in snippet.strip().splitlines(True) if s.strip()])
            if max(resultchecks) > .5 or show_all:
                yield (
                    'result',
                    (
                        str(file_path),
                        percent,
                        admin_desc,
                        enduser_desc,
                        chatgpt_conf_percent,
                        snippet,
                    )
                )
        yield ('progress', index + 1, len(file_list))


def run_scan(scan_path: str, deep_scan: bool, show_all: bool, use_gpt: bool, cancel_event: threading.Event) -> None:
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
    """
    try:
        for event_type, data in scan_files(scan_path, deep_scan, show_all, use_gpt, cancel_event):
            if cancel_event.is_set():
                break
            if event_type == 'progress':
                current, total = data[1], data[2]
                if current == 0:
                    enqueue_ui_update(configure_progress, total)
                else:
                    enqueue_ui_update(update_progress, current)
            elif event_type == 'result':
                enqueue_ui_update(insert_tree_row, data)
    finally:
        enqueue_ui_update(finish_scan_state)


def run_cli(path: str, deep: bool, show_all: bool, use_gpt: bool) -> None:
    """Run scans and stream results to stdout as CSV rows.

    Parameters
    ----------
    path : str
        Directory to scan.
    deep : bool
        Whether to evaluate all 1024-byte windows.
    show_all : bool
        Whether to emit every scanned file.
    use_gpt : bool
        Whether to request GPT analysis for confident detections.
    """
    writer = csv.writer(sys.stdout)
    writer.writerow(("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"))

    cancel_event = threading.Event()

    for event_type, data in scan_files(path, deep, show_all, use_gpt, cancel_event):
        if event_type == 'result':
            writer.writerow(data)


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


def create_gui() -> tk.Tk:
    """Construct and return the main Tkinter GUI for the scanner.

    Returns
    -------
    tk.Tk
        Initialized Tk root instance ready for ``mainloop``.
    """
    global root, textbox, progress_bar, deep_var, all_var, gpt_var, tree, scan_button, cancel_button

    root = tk.Tk()
    root.geometry("800x500")
    root.title("GPT Virus Scanner")

    label = tk.Label(root, text="Path to scan")
    label.pack()

    textbox = tk.Entry(root)
    textbox.pack()
    select_dir_btn = tk.Button(root, text="Select Directory", command=browse_button_click)
    select_dir_btn.pack()
    deep_var = tk.BooleanVar()
    deep_checkbox = tk.Checkbutton(root, text="Deep scan", variable=deep_var)
    deep_checkbox.pack()

    all_var = tk.BooleanVar()
    all_checkbox = tk.Checkbutton(root, text="Show all files", variable=all_var)
    all_checkbox.pack()

    gpt_var = tk.BooleanVar()
    gpt_checkbox = tk.Checkbutton(root, text="Use ChatGPT", variable=gpt_var)
    if not Config.GPT_ENABLED:
        gpt_var.set(False)
        gpt_checkbox.config(state="disabled")
        messagebox.showwarning("GPT Disabled",
                                       "apikey.txt or task.txt not found. GPT functionality is disabled.")

    gpt_checkbox.pack()

    if Config.extensions_missing:
        default_exts = ', '.join(sorted(Config.extensions_set)) if Config.extensions_set else 'none'
        messagebox.showwarning(
            "Extensions Missing",
            f"extensions.txt not found. Using default extensions: {default_exts}"
        )

    scan_button = tk.Button(root, text="Scan now", command=button_click)
    scan_button.pack()
    cancel_button = tk.Button(root, text="Cancel", command=cancel_scan, state="disabled")
    cancel_button.pack()
    export_button = tk.Button(root, text="Export CSV", command=export_results)
    export_button.pack()
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
    progress_bar.pack()
    style = ttk.Style(root)
    style.configure('Scanner.Treeview', rowheight=100)
    tree = ttk.Treeview(root, style='Scanner.Treeview')
    tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("path", width=200, stretch=tk.NO, anchor="w")
    tree.column("own_conf", width=50, stretch=tk.NO, anchor="e")
    tree.column("admin_desc", width=200, stretch=tk.NO, anchor="w")
    tree.column("end-user_desc", width=200, stretch=tk.NO, anchor="w")
    tree.column("gpt_conf", width=50, stretch=tk.NO, anchor="e")
    tree.column("snippet", width=50, stretch=tk.NO, anchor="w")
    root.after(0, process_ui_queue)

    tree.heading("#0", text="")
    tree.heading("path", text="File Path", command=lambda: sort_column(tree, "path", False))
    tree.heading("own_conf", text="Own confidence",
                 command=lambda: sort_column(tree, "own_conf", False))
    tree.heading("admin_desc", text="Administrator Description",
                 command=lambda: sort_column(tree, "admin_desc", False))
    tree.heading("end-user_desc", text="End-User Description",
                 command=lambda: sort_column(tree, "end-user_desc", False))
    tree.heading("gpt_conf", text="ChatGPT confidence",
                 command=lambda: sort_column(tree, "gpt_conf", False))
    tree.heading("snippet", text="Snippet", command=lambda: sort_column(tree, "snippet", False))

    tree.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tree.configure(yscrollcommand=scrollbar.set)
    tree.bind('<ButtonRelease-1>', partial(motion_handler, tree))
    motion_handler(tree, None)   # Perform initial wrapping
    set_scanning_state(False)
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPT Virus Scanner")
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    parser.add_argument('--path', type=str, help='Directory to scan')
    parser.add_argument('--deep', action='store_true', help='Perform a deep scan')
    parser.add_argument('--show-all', action='store_true', help='Show all files in the output')
    parser.add_argument('--use-gpt', action='store_true', help='Use GPT for analysis')
    parser.add_argument(
        '--extensions',
        type=str,
        help='Comma-separated list of file extensions to scan (overrides extensions.txt)'
    )
    args = parser.parse_args()

    if args.extensions:
        extension_list = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
        Config.set_extensions(extension_list, missing=False)

    if args.cli:
        if not args.path:
            parser.error('--path is required in CLI mode')
        run_cli(args.path, args.deep, args.show_all, args.use_gpt)
    else:
        app_root = create_gui()
        app_root.mainloop()
