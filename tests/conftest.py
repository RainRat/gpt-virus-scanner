import sys
import pytest
from unittest.mock import MagicMock

# Create a mock object for tkinter
mock_tk = MagicMock()

# Explicitly mock attributes accessed during import or usage
mock_tk.Tk = MagicMock
mock_tk.Label = MagicMock
mock_tk.Entry = MagicMock
mock_tk.Button = MagicMock
mock_tk.BooleanVar = MagicMock
mock_tk.Checkbutton = MagicMock
mock_tk.END = "end"
mock_tk.BOTH = "both"
mock_tk.NO = "no"
mock_tk.RIGHT = "right"
mock_tk.Y = "y"
mock_tk.Event = MagicMock
mock_tk.TclError = Exception

# Inject into sys.modules
sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.font'] = MagicMock()

# Explicitly mock Label to be a real class so it's not a mock instance
class MockLabel:
    def __init__(self, *args, **kwargs): pass
    def grid(self, *args, **kwargs): pass
    def pack(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass
    def cget(self, *args, **kwargs): return ""
    def __repr__(self): return "MockLabel"

mock_tk.Label = MockLabel

mock_ttk = MagicMock()
# Explicitly mock Frame to be a real class so it's not a mock instance
class MockFrame:
    def __init__(self, *args, **kwargs): pass
    def grid(self, *args, **kwargs): pass
    def pack(self, *args, **kwargs): pass
    def columnconfigure(self, *args, **kwargs): pass
    def rowconfigure(self, *args, **kwargs): pass
    def winfo_viewable(self): return True
    def __repr__(self): return "MockFrame"

mock_ttk.Frame = MockFrame
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.scrolledtext'] = MagicMock()

@pytest.fixture(autouse=True)
def reset_globals():
    import gptscan
    gptscan.current_cancel_event = None
    gptscan._all_results_cache = []
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
