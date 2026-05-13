import sys
import pytest
from unittest.mock import MagicMock

# Define real functions for mainloop to satisfy matplotlib's check
def mock_mainloop(*args, **kwargs):
    pass

class MockMisc:
    @staticmethod
    def mainloop(*args, **kwargs):
        pass

class MockWidget:
    def __init__(self, *args, **kwargs):
        self.master = args[0] if args else None
        self.children = []
    def grid(self, *args, **kwargs): return self
    def grid_forget(self, *args, **kwargs): pass
    def grid_remove(self, *args, **kwargs): pass
    def pack(self, *args, **kwargs): return self
    def pack_forget(self, *args, **kwargs): pass
    def place(self, *args, **kwargs): return self
    def place_forget(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass
    def configure(self, *args, **kwargs): pass
    def cget(self, *args, **kwargs): return ""
    def delete(self, *args, **kwargs): pass
    def insert(self, *args, **kwargs): pass
    def bind(self, *args, **kwargs): pass
    def unbind(self, *args, **kwargs): pass
    def winfo_viewable(self): return True
    def winfo_children(self): return self.children
    def destroy(self): pass
    def update(self): pass
    def update_idletasks(self): pass
    def __setitem__(self, key, value): pass
    def __getitem__(self, key): return MagicMock()
    def __repr__(self): return f"Mock{self.__class__.__name__}"

# Explicitly mock basic components as real classes to avoid metaclass conflicts
class MockFrame(MockWidget):
    def columnconfigure(self, *args, **kwargs): pass
    def rowconfigure(self, *args, **kwargs): pass

class MockTk(MockFrame):
    def withdraw(self): pass
    def deiconify(self): pass
    def title(self, *args): pass
    def geometry(self, *args): pass
    def protocol(self, *args): pass
    def mainloop(self, *args): pass
    def quit(self): pass

class MockLabel(MockWidget):
    pass

class MockPanedwindow(MockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panes = []
    def add(self, child, **kwargs):
        if child not in self._panes:
            self._panes.append(child)
    def insert(self, pos, child, **kwargs):
        if child not in self._panes:
            self._panes.insert(pos, child)
    def forget(self, child):
        if child in self._panes:
            self._panes.remove(child)
    def panes(self):
        return self._panes

class MockMenu(MockWidget):
    def add_command(self, *args, **kwargs): pass
    def add_separator(self, *args, **kwargs): pass
    def add_cascade(self, *args, **kwargs): pass
    def delete(self, *args, **kwargs): pass
    def entryconfig(self, *args, **kwargs): pass
    def post(self, *args, **kwargs): pass
    def unpost(self, *args, **kwargs): pass

# A custom mock class that returns MagicMock for any attribute not explicitly set
class MockTkModule:
    def __init__(self):
        self.mainloop = mock_mainloop
        self.Misc = MockMisc
        self.Tk = MockTk
        self.Toplevel = MockTk
        self.Frame = MockFrame
        self.Label = MockLabel
        self.Entry = MagicMock
        self.Button = MagicMock
        self.Checkbutton = MagicMock
        self.BooleanVar = MagicMock
        self.StringVar = MagicMock
        self.IntVar = MagicMock
        self.Menu = MockMenu
        self.Text = MagicMock
        self.Scrollbar = MagicMock
        self.TclError = Exception
        self.END = "end"
        self.BOTH = "both"
        self.Y = "y"
        self.X = "x"
        self.NO = "no"
        self.YES = "yes"
        self.RIGHT = "right"
        self.LEFT = "left"
        self.TOP = "top"
        self.BOTTOM = "bottom"
        self.NONE = "none"
        self.VERTICAL = "vertical"
        self.HORIZONTAL = "horizontal"
        self.Event = MagicMock
        self.messagebox = MagicMock()
        self.filedialog = MagicMock()
        self.simpledialog = MagicMock()
        self.ttk = MagicMock()
        self.font = MagicMock()
        self.scrolledtext = MagicMock()
        
        # Populate ttk with components
        self.ttk.Frame = MockFrame
        self.ttk.Label = MockLabel
        self.ttk.Button = MagicMock
        self.ttk.Entry = MagicMock
        self.ttk.Treeview = MagicMock
        self.ttk.Separator = MockWidget
        self.ttk.Panedwindow = MockPanedwindow
        self.ttk.Menubutton = MockWidget
        self.ttk.Progressbar = MockWidget
        self.ttk.Spinbox = MockWidget
        self.ttk.Combobox = MockWidget

    def __getattr__(self, name):
        return MagicMock()

mock_tk = MockTkModule()

# Inject into sys.modules
sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.messagebox'] = mock_tk.messagebox
sys.modules['tkinter.filedialog'] = mock_tk.filedialog
sys.modules['tkinter.simpledialog'] = mock_tk.simpledialog
sys.modules['tkinter.ttk'] = mock_tk.ttk
sys.modules['tkinter.font'] = mock_tk.font
sys.modules['tkinter.scrolledtext'] = mock_tk.scrolledtext

@pytest.fixture(autouse=True)
def reset_globals():
    import gptscan
    gptscan.current_cancel_event = None
    gptscan._all_results_cache = []
    gptscan._last_scan_summary = ""
    gptscan._virtual_source_cache = {}
    gptscan._model_cache = None
    gptscan._async_openai_client = None

    # Save original Config state
    orig_exts = gptscan.Config.extensions_set.copy()
    orig_threshold = gptscan.Config.THRESHOLD
    orig_max_file_size = gptscan.Config.MAX_FILE_SIZE
    orig_max_source_view_size = gptscan.Config.MAX_SOURCE_VIEW_SIZE

    yield

    # Restore original Config state
    gptscan.Config.extensions_set = orig_exts
    gptscan.Config.THRESHOLD = orig_threshold
    gptscan.Config.MAX_FILE_SIZE = orig_max_file_size
    gptscan.Config.MAX_SOURCE_VIEW_SIZE = orig_max_source_view_size

@pytest.fixture
def mock_tf_env(monkeypatch):
    import gptscan
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.5]]
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    mock_tf = MagicMock()
    mock_tf.constant = lambda x: x
    mock_tf.expand_dims = lambda x, axis: x
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    return mock_model
