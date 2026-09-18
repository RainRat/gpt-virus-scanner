import inspect
import gptscan


def test_ui_settings_frame_layout_source():
    """Verify that settings_frame separates options_frame and provider_frame into boxes_frame,

    and positions copy_cmd_button in a dedicated action bar (cmd_bar).
    """
    source = inspect.getsource(gptscan.create_gui)

    # Verify boxes_frame container exists
    assert 'boxes_frame = ttk.Frame(settings_frame)' in source
    assert 'boxes_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)' in source

    # Verify options_frame and provider_frame are packed into boxes_frame
    assert 'options_frame = ttk.LabelFrame(boxes_frame, text="Scan Options", padding=10)' in source
    assert 'provider_frame = ttk.LabelFrame(boxes_frame, text="AI Analysis", padding=10)' in source

    # Verify cmd_bar action bar exists and contains copy_cmd_button
    assert 'cmd_bar = ttk.Frame(settings_frame)' in source
    assert 'copy_cmd_button = ttk.Button(cmd_bar, text="Copy CLI Command", command=copy_cli_command)' in source
