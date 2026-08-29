from gptscan import export_results

def test_export_results_docstring_lists_all_formats():
    doc = export_results.__doc__
    assert doc is not None
    assert "CSV" in doc
    assert "Markdown" in doc
    assert "HTML" in doc
    assert "JSON" in doc
    assert "YAML" in doc
    assert "SARIF" in doc
    assert "XML" in doc
    assert "Console Triage Report" in doc
