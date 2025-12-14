import sys
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
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
