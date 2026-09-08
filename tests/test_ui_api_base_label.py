import inspect
import gptscan

def test_api_base_url_label_in_gui_source():
    """Verify that the API base endpoint label in create_gui is titled 'API Base URL:'."""
    source = inspect.getsource(gptscan.create_gui)
    assert 'ttk.Label(provider_frame, text="API Base URL:")' in source
    assert 'API Base Web Link:' not in source
