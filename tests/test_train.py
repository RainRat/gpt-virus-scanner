import pytest
import numpy as np
import yaml
import tensorflow as tf
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from train import Hyperparameters, DataLoader, ModelBuilder, ModelConfig, load_config, Trainer, Predictor

def test_hyperparameters_to_list():
    hp = Hyperparameters(
        embedding_scale=0.1, rnn_scale=0.2, pooling_type=0.3,
        dropout1=0.4, dense_scale=0.5, activation=0.6, dropout2=0.7,
        spatial_dropout=0.8, rnn_type=0.9, use_conv=0.1,
        conv_filters_scale=0.2, conv_padding=0.3, kernel_init=0.4,
        rnn_dropout=0.5, rnn_recurrent_dropout=0.6, conv_kernel_scale=0.7,
        optimizer=0.8
    )
    lst = hp.to_list()
    assert len(lst) == 17
    assert lst[0] == 0.1
    assert lst[-1] == 0.8

def test_hyperparameters_from_list():
    lst = [0.1] * 17
    hp = Hyperparameters.from_list(lst)
    assert hp.embedding_scale == 0.1
    assert hp.optimizer == 0.1

def test_hyperparameters_mutate():
    hp = Hyperparameters(*([0.5] * 17))
    mutated = hp.mutate()
    assert hp != mutated
    diffs = [h != m for h, m in zip(hp.to_list(), mutated.to_list())]
    assert any(diffs)

def test_hyperparameters_get_derived_params():
    # Test with scale 0.0
    hp_min = Hyperparameters(*([0.0] * 17))
    dp_min = hp_min.get_derived_params()
    assert dp_min['embedding_dim'] == 32
    assert dp_min['rnn_units'] == 32
    assert dp_min['dense_units'] == 32
    assert dp_min['conv_filters'] == 8
    assert dp_min['conv_kernel_size'] == 2
    assert dp_min['spatial_dropout'] == 0.0
    assert dp_min['rnn_dropout'] == 0.0
    assert dp_min['rnn_recurrent_dropout'] == 0.0
    assert dp_min['dropout1'] == 0.0
    assert dp_min['dropout2'] == 0.0

    # Test with scale 1.0
    hp_max = Hyperparameters(*([1.0] * 17))
    dp_max = hp_max.get_derived_params()
    assert dp_max['embedding_dim'] == 160
    assert dp_max['rnn_units'] == 160
    assert dp_max['dense_units'] == 160
    assert dp_max['conv_filters'] == 98
    assert dp_max['conv_kernel_size'] == 5
    assert dp_max['spatial_dropout'] == 0.5
    assert dp_max['rnn_dropout'] == 0.5
    assert dp_max['rnn_recurrent_dropout'] == 0.5
    assert dp_max['dropout1'] == 0.5
    assert dp_max['dropout2'] == 0.5

    # Test with scale 0.5
    hp_mid = Hyperparameters(*([0.5] * 17))
    dp_mid = hp_mid.get_derived_params()
    # int(0.5 * 128) + 32 = 64 + 32 = 96
    assert dp_mid['embedding_dim'] == 96
    # int(0.5 * 90) + 8 = 45 + 8 = 53
    assert dp_mid['conv_filters'] == 53
    # int(0.5 * 3) + 2 = 1 + 2 = 3
    assert dp_mid['conv_kernel_size'] == 3
    assert dp_mid['dropout1'] == 0.25

def test_dataloader_load_file_padding(tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    loader = DataLoader(config)

    file_path = tmp_path / "small.bin"
    file_path.write_bytes(b"\x01\x02\x03")

    data = loader.load_file(file_path)
    assert len(data) == 10
    assert data[:3] == [1, 2, 3]
    assert data[3:] == [0] * 7

def test_dataloader_load_file_truncation(tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    loader = DataLoader(config)

    file_path = tmp_path / "large.bin"
    file_path.write_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C")

    data = loader.load_file(file_path)
    # The corrected implementation should take half from start and half from end.
    # max_length = 10. half = 5. other_half = 5.
    # first 5: 1, 2, 3, 4, 5.
    # last 5: 8, 9, 10, 11, 12.
    assert len(data) == 10
    assert data == [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]

def test_dataloader_load_dataset(tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=2.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    loader = DataLoader(config)

    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    (pos_dir / "1.bin").write_bytes(b"\x01")

    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    (neg_dir / "0.bin").write_bytes(b"\x00")

    x, y, sw = loader.load_dataset(pos_dir, neg_dir)
    assert len(x) == 2
    assert y[0] == 1
    assert y[1] == 0
    assert sw[0] == 2.0
    assert sw[1] == 1.0

def test_dataloader_load_prediction_files(tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    loader = DataLoader(config)

    input_dir = tmp_path / "predict"
    input_dir.mkdir()
    (input_dir / "b.bin").write_bytes(b"\x02")
    (input_dir / "a.bin").write_bytes(b"\x01")

    paths, data = loader.load_prediction_files(input_dir)
    assert len(paths) == 2
    assert paths[0].name == "a.bin"
    assert paths[1].name == "b.bin"
    assert data[0][0] == 1
    assert data[1][0] == 2

def test_model_builder_mappings():
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    builder = ModelBuilder(config)

    assert builder._get_activation(0.1) == "relu"
    assert builder._get_activation(0.4) == "sigmoid"
    assert builder._get_activation(0.6) == "tanh"
    assert builder._get_activation(0.9) == "hard_sigmoid"

    assert builder._get_optimizer(0.1) == "sgd"
    assert builder._get_optimizer(0.3) == "rmsprop"
    assert builder._get_optimizer(0.5) == "adam"
    assert builder._get_optimizer(0.7) == "adagrad"
    assert builder._get_optimizer(0.9) == "nadam"

    assert builder._get_pooling_type(0.8) == "avg"
    assert builder._get_pooling_type(0.6) == "max"
    assert builder._get_pooling_type(0.3) == "both"
    assert builder._get_pooling_type(0.1) == "none"

    # Coverage for _get_initializer loop and final return
    assert builder._get_initializer(0.05) == "glorot_normal"
    assert builder._get_initializer(0.95) == "random_uniform"

def test_model_builder_build_model():
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    builder = ModelBuilder(config)

    # Test different RNN types
    for rnn_type in [0.1, 0.4, 0.6, 0.9]:
        hp = Hyperparameters(*([0.5] * 17))
        hp.rnn_type = rnn_type
        model = builder.build_model(hp)
        assert model is not None
        assert isinstance(model, tf.keras.Model)

    # Test with Conv1D
    hp = Hyperparameters(*([0.5] * 17))
    hp.use_conv = 0.6
    model = builder.build_model(hp)
    assert model is not None

def test_model_builder_too_large(capsys):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=10,
        pad_value=0
    )
    builder = ModelBuilder(config)
    hp = Hyperparameters(*([0.5] * 17))
    model = builder.build_model(hp)
    assert model is None
    out, _ = capsys.readouterr()
    assert "Model too large" in out

def test_model_builder_print_architecture(capsys):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    builder = ModelBuilder(config)
    hp = Hyperparameters(*([0.5] * 17))
    hp.use_conv = 0.6
    builder.print_architecture(hp)
    out, _ = capsys.readouterr()
    assert "Features" in out
    assert "Conv1D" in out

def test_load_config(tmp_path):
    config_data = {
        'model': {
            'name': 'test_model',
            'max_length': 1024,
            'pad_value': 0,
            'max_params': 1000000
        },
        'training': {
            'batch_size': 32,
            'epochs': 10,
            'validation_split': 0.2,
            'patience': 5,
            'mode': 'train'
        },
        'prediction': {
            'threshold': 0.5
        },
        'weights': {
            'positive_sample_weight': 2.0
        },
        'hyperparameters': {
            'embedding_scale': 0.5, 'rnn_scale': 0.5, 'pooling_type': 0.5,
            'dropout1': 0.5, 'dense_scale': 0.5, 'activation': 0.5, 'dropout2': 0.5,
            'spatial_dropout': 0.5, 'rnn_type': 0.5, 'use_conv': 0.5,
            'conv_filters_scale': 0.5, 'conv_padding': 0.5, 'kernel_init': 0.5,
            'rnn_dropout': 0.5, 'rnn_recurrent_dropout': 0.5, 'conv_kernel_scale': 0.5,
            'optimizer': 0.5
        }
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    config, hp = load_config(str(config_file))
    assert config.model_name == 'test_model'
    assert config.max_length == 1024
    assert hp.embedding_scale == 0.5

def test_trainer_save_load_hp(tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="train", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )
    trainer = Trainer(config)
    hp = Hyperparameters(*([0.5] * 17))
    trainer.best_hp = hp
    trainer.best_acc = 0.9
    trainer.best_loss = 0.1

    hp_file = tmp_path / "hp.yml"
    trainer.save_hyperparameters(str(hp_file))

    loaded_hp = Trainer.load_hyperparameters(str(hp_file))
    assert loaded_hp.embedding_scale == 0.5

    with open(hp_file, 'r') as f:
        data = yaml.safe_load(f)
        assert data['metrics']['best_accuracy'] == 0.9

@patch("tensorflow.keras.models.load_model")
def test_predictor_predict(mock_load_model, tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="predict", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "test.bin").write_bytes(b"\x01" * 10)

    output_dir = tmp_path / "output"

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8]]) # Above threshold
    mock_load_model.return_value = mock_model

    predictor = Predictor(config)
    predictor.predict("dummy_path", input_dir, output_dir)

    assert (output_dir / "test.bin").exists()

@patch("tensorflow.keras.models.load_model")
def test_predictor_predict_below_threshold(mock_load_model, tmp_path):
    config = ModelConfig(
        model_name="test", max_length=10, batch_size=32, epochs=1,
        mode="predict", predict_threshold=0.5, positive_sample_weight=1.0,
        validation_split=0.2, patience=3,
        max_params=1000000, pad_value=0
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "test.bin").write_bytes(b"\x01" * 10)

    output_dir = tmp_path / "output"

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.3]]) # Below threshold
    mock_load_model.return_value = mock_model

    predictor = Predictor(config)
    predictor.predict("dummy_path", input_dir, output_dir)

    assert not (output_dir / "test.bin").exists()
