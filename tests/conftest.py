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
    yield
