import sys
from unittest.mock import MagicMock
import pytest
import gptscan


def test_get_model_cached(monkeypatch):
    monkeypatch.setattr(gptscan, "_model_cache", "cached_model_instance")
    result = gptscan.get_model()
    assert result == "cached_model_instance"


def test_get_model_primary_success(monkeypatch):
    monkeypatch.setattr(gptscan, "_model_cache", None)
    mock_tf = MagicMock()
    mock_model = MagicMock()
    mock_tf.keras.models.load_model.return_value = mock_model
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    result = gptscan.get_model()

    assert result == mock_model
    mock_tf.keras.models.load_model.assert_called_once_with('scripts.h5', compile=False)
    assert gptscan._model_cache == mock_model


def test_get_model_fallback_tf_keras(monkeypatch):
    monkeypatch.setattr(gptscan, "_model_cache", None)
    mock_tf = MagicMock()
    mock_tf.keras.models.load_model.side_effect = ValueError("Primary load failed")
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)

    mock_tf_keras = MagicMock()
    fallback_model = MagicMock()
    mock_tf_keras.models.load_model.return_value = fallback_model
    monkeypatch.setitem(sys.modules, 'tf_keras', mock_tf_keras)

    result = gptscan.get_model()

    assert result == fallback_model
    mock_tf_keras.models.load_model.assert_called_once_with('scripts.h5', compile=False)
    assert gptscan._model_cache == fallback_model
