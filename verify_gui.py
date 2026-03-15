import tkinter as tk
import gptscan
import threading
import time
import os
import signal

def run_app():
    root = gptscan.create_gui()
    # Close after 5 seconds to give time for screenshot
    root.after(5000, root.destroy)
    root.mainloop()

if __name__ == "__main__":
    # Start app in a separate thread or just run it and hope for the best
    # xvfb-run will handle the display
    run_app()
