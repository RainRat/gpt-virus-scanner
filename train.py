import tensorflow as tf
import numpy as np
import yaml
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from tensorflow.keras.layers import (
    Dense, Input, LSTM, Embedding, Dropout, GRU, concatenate,
    Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, 
    SpatialDropout1D, Conv1D, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session, count_params


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    model_name: str
    max_length: int
    batch_size: int
    epochs: int
    mode: str
    predict_threshold: float
    positive_sample_weight: float
    positive_class_weight: float
    validation_split: float
    patience: int
    max_params: int
    pad_value: int


@dataclass
class Hyperparameters:
    """Settings that control how the AI 'brain' is built and trained."""
    embedding_scale: float
    rnn_scale: float
    pooling_type: float
    dropout1: float
    dense_scale: float
    activation: float
    dropout2: float
    spatial_dropout: float
    rnn_type: float
    use_conv: float
    conv_filters_scale: float
    conv_padding: float
    kernel_init: float
    rnn_dropout: float
    rnn_recurrent_dropout: float
    conv_kernel_scale: float
    optimizer: float

    def to_list(self) -> List[float]:
        """Convert to list format for compatibility."""
        return [
            self.embedding_scale, self.rnn_scale, self.pooling_type,
            self.dropout1, self.dense_scale, self.activation, self.dropout2,
            self.spatial_dropout, self.rnn_type, self.use_conv,
            self.conv_filters_scale, self.conv_padding, self.kernel_init,
            self.rnn_dropout, self.rnn_recurrent_dropout, self.conv_kernel_scale,
            self.optimizer
        ]

    @classmethod
    def from_list(cls, params: List[float]) -> 'Hyperparameters':
        """Create from list format."""
        return cls(*params)

    def mutate(self) -> 'Hyperparameters':
        """Create a mutated copy of hyperparameters."""
        params = self.to_list()
        params[random.randint(0, 16)] = random.random()
        params[random.randint(0, 16)] = random.random()
        return Hyperparameters.from_list(params)


class DataLoader:
    """Handles loading and preprocessing of binary file data."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_length = config.max_length
        self.pad_value = config.pad_value
    
    def load_file(self, file_path: Path) -> List[int]:
        """Load and preprocess a single file."""
        with open(file_path, 'rb') as f:
            data = list(f.read(self.max_length))
        
        # Apply padding logic
        if len(data) < self.max_length:
            num_to_add = self.max_length - len(data)
            data.extend([self.pad_value] * num_to_add)
        elif len(data) > self.max_length:
            # Take half from start and half from end
            half = self.max_length // 2
            data = data[:half] + data[-half:]
        
        return data
    
    def load_dataset(self, positive_dir: Path, negative_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load complete dataset with sample weights."""
        x_train = []
        y_train = []
        sample_weights = []
        
        # Load positive examples
        for file_path in positive_dir.iterdir():
            if file_path.is_file():
                data = self.load_file(file_path)
                x_train.append(data)
                y_train.append(1)
                sample_weights.append(self.config.positive_sample_weight)
        
        # Load negative examples
        for file_path in negative_dir.iterdir():
            if file_path.is_file():
                data = self.load_file(file_path)
                x_train.append(data)
                y_train.append(0)
                sample_weights.append(1.0)
        
        return (
            np.array(x_train),
            np.array(y_train),
            np.array(sample_weights)
        )
    
    def load_prediction_files(self, directory: Path) -> Tuple[List[Path], np.ndarray]:
        """Load files for prediction."""
        file_paths = sorted([f for f in directory.iterdir() if f.is_file()])
        data = np.array([self.load_file(f) for f in file_paths])
        return file_paths, data


class ModelBuilder:
    """Builds neural network models based on hyperparameters."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_length = config.max_length
    
    def _get_activation(self, value: float) -> str:
        """Map hyperparameter to activation function."""
        if value < 0.25:
            return "relu"
        elif value < 0.5:
            return "sigmoid"
        elif value < 0.75:
            return "tanh"
        else:
            return "hard_sigmoid"
    
    def _get_initializer(self, value: float) -> str:
        """Map hyperparameter to kernel initializer."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        initializers = [
            "glorot_normal", "glorot_uniform", "he_normal", "he_uniform",
            "lecun_normal", "lecun_uniform", "truncated_normal",
            "orthogonal", "random_normal", "random_uniform"
        ]
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return initializers[i]
        return initializers[-1]
    
    def _get_optimizer(self, value: float) -> str:
        """Map hyperparameter to optimizer."""
        if value < 0.2:
            return "sgd"
        elif value < 0.4:
            return "rmsprop"
        elif value < 0.6:
            return "adam"
        elif value < 0.8:
            return "adagrad"
        else:
            return "nadam"
    
    def _get_pooling_type(self, value: float) -> str:
        """Map hyperparameter to pooling type."""
        if value > 0.75:
            return "avg"
        elif value > 0.50:
            return "max"
        elif value > 0.25:
            return "both"
        else:
            return "none"
    
    def build_model(self, hp: Hyperparameters) -> Optional[Model]:
        """Build model from hyperparameters."""
        try:
            # Calculate integer parameters
            embedding_dim = int(hp.embedding_scale * 128) + 32
            rnn_units = int(hp.rnn_scale * 128) + 32
            dense_units = int(hp.dense_scale * 128) + 32
            conv_filters = int(hp.conv_filters_scale * 90) + 8
            conv_kernel_size = int(hp.conv_kernel_scale * 3) + 2
            
            # Get categorical parameters
            activation = self._get_activation(hp.activation)
            initializer = self._get_initializer(hp.kernel_init)
            optimizer = self._get_optimizer(hp.optimizer)
            pooling = self._get_pooling_type(hp.pooling_type)
            
            # Build model
            inp = Input(shape=(self.max_length,))
            x = Embedding(256, embedding_dim)(inp)
            x = SpatialDropout1D(hp.spatial_dropout * 0.5)(x)
            
            # RNN layer selection
            if hp.rnn_type > 0.75:
                x = LSTM(
                    rnn_units, return_sequences=True,
                    dropout=hp.rnn_dropout * 0.5,
                    recurrent_dropout=hp.rnn_recurrent_dropout * 0.5
                )(x)
            elif hp.rnn_type > 0.5:
                x = GRU(
                    rnn_units, return_sequences=True,
                    dropout=hp.rnn_dropout * 0.5,
                    recurrent_dropout=hp.rnn_recurrent_dropout * 0.5
                )(x)
            elif hp.rnn_type > 0.25:
                x = Bidirectional(GRU(
                    rnn_units, return_sequences=True,
                    dropout=hp.rnn_dropout * 0.5,
                    recurrent_dropout=hp.rnn_recurrent_dropout * 0.5
                ))(x)
            else:
                x = Bidirectional(LSTM(
                    rnn_units, return_sequences=True,
                    dropout=hp.rnn_dropout * 0.5,
                    recurrent_dropout=hp.rnn_recurrent_dropout * 0.5
                ))(x)
            
            # Optional Conv1D layer
            if hp.use_conv > 0.5:
                padding = 'same' if hp.conv_padding < 0.5 else 'valid'
                x = Conv1D(
                    conv_filters, kernel_size=conv_kernel_size,
                    padding=padding, kernel_initializer=initializer
                )(x)
            
            # Pooling layer selection
            if pooling == "avg":
                x = GlobalAveragePooling1D()(x)
            elif pooling == "max":
                x = GlobalMaxPool1D()(x)
            elif pooling == "both":
                avg_pool = GlobalAveragePooling1D()(x)
                max_pool = GlobalMaxPool1D()(x)
                x = concatenate([avg_pool, max_pool])
            else:
                x = Flatten()(x)
            
            # Dense layers
            x = Dropout(hp.dropout1 * 0.5)(x)
            x = Dense(dense_units, activation=activation)(x)
            x = Dropout(hp.dropout2 * 0.5)(x)
            x = Dense(1, activation="sigmoid")(x)
            
            # Compile model
            model = Model(inputs=inp, outputs=x)
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'],
                weighted_metrics=['accuracy', 'binary_crossentropy']
            )
            
            # Check model size
            trainable_count = np.sum([count_params(w) for w in model.trainable_weights])
            non_trainable_count = np.sum([count_params(w) for w in model.non_trainable_weights])
            
            if (trainable_count + non_trainable_count) > self.config.max_params:
                print(f"Model too large: {trainable_count + non_trainable_count} parameters")
                clear_session()
                return None
            
            return model
            
        except Exception as e:
            print(f"Failed to compile model: {e}")
            clear_session()
            return None
    
    def print_architecture(self, hp: Hyperparameters):
        """Print human-readable architecture description."""
        rnn_type = "BiLSTM" if hp.rnn_type < 0.25 else \
                   "BiGRU" if hp.rnn_type < 0.5 else \
                   "GRU" if hp.rnn_type < 0.75 else "LSTM"
        
        pooling = self._get_pooling_type(hp.pooling_type)
        
        print(f"Features: {int(hp.embedding_scale * 128) + 32}, "
              f"SpatialDrop: {hp.spatial_dropout * 0.5:.3f}, "
              f"{rnn_type} Size: {int(hp.rnn_scale * 128) + 32}, "
              f"Drop: {hp.rnn_dropout * 0.5:.3f}, "
              f"RecDrop: {hp.rnn_recurrent_dropout * 0.5:.3f}")
        
        if hp.use_conv > 0.5:
            padding = 'same' if hp.conv_padding < 0.5 else 'valid'
            print(f"Conv1D Filt: {int(hp.conv_filters_scale * 90) + 8}, "
                  f"Pad: {padding}, "
                  f"Init: {self._get_initializer(hp.kernel_init)}, "
                  f"Size: {int(hp.conv_kernel_scale * 3) + 2}")
        
        print(f"Pool: {pooling}, "
              f"Dropout1: {hp.dropout1 * 0.5:.3f}, "
              f"Dense: {int(hp.dense_scale * 128) + 32}, "
              f"Activation: {self._get_activation(hp.activation)}, "
              f"Dropout2: {hp.dropout2 * 0.5:.3f}, "
              f"Opt: {self._get_optimizer(hp.optimizer)}")


class Trainer:
    """Handles model training with genetic algorithm optimization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_builder = ModelBuilder(config)
        self.data_loader = DataLoader(config)
        self.best_hp: Optional[Hyperparameters] = None
        self.best_loss = 9999.0
        self.best_acc = 0.0
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              sample_weights: np.ndarray, initial_hp: Optional[Hyperparameters] = None):
        """Train models using genetic algorithm optimization."""
        current_hp = initial_hp
        if current_hp is None:
            raise ValueError("No initial hyperparameters provided. Load from config or provide starting values.")
        
        self.best_hp = current_hp
        should_mutate = False
        
        while True:
            # Mutate hyperparameters
            if should_mutate:
                current_hp = self.best_hp.mutate()
            should_mutate = True
            
            # Build model
            self.model_builder.print_architecture(current_hp)
            model = self.model_builder.build_model(current_hp)
            
            if model is None:
                continue
            
            # Train model
            model.summary()
            early_stop = EarlyStopping(
                monitor='val_weighted_binary_crossentropy',
                min_delta=0.0001,
                patience=self.config.patience,
                verbose=1,
                restore_best_weights=True
            )
            
            history = model.fit(
                x_train, y_train,
                sample_weight=sample_weights,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                shuffle=True,
                callbacks=[early_stop]
            )
            
            # Evaluate results
            val_acc = history.history['val_weighted_acc']
            val_loss = history.history['val_weighted_binary_crossentropy']
            
            print("\nCumulative:")
            print(f"Accuracy Best: {self.best_acc:.4f}")
            print(f"Loss Best: {self.best_loss:.4f}")
            
            print("\nThis Round:")
            print(f"Accuracy Final: {val_acc[-1]:.4f}")
            print(f"Accuracy Best: {max(val_acc):.4f}")
            print(f"Loss Final: {val_loss[-1]:.4f}")
            print(f"Loss Best: {min(val_loss):.4f}")
            
            # Save if improved
            if min(val_loss) < self.best_loss:
                print("\n" + "*" * 50)
                print("New Best Model!")
                print("*" * 50 + "\n")
                
                self.best_hp = current_hp
                self.best_loss = min(val_loss)
                self.best_acc = max(val_acc)
                
                # Save model and hyperparameters
                model.save(f'{self.config.model_name}.h5')
                self.save_hyperparameters(f'{self.config.model_name}_best_hp.yml')
            
            # Cleanup
            clear_session()
    
    def save_hyperparameters(self, filepath: str):
        """Save best hyperparameters to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump({
                'hyperparameters': asdict(self.best_hp),
                'metrics': {
                    'best_accuracy': float(self.best_acc),
                    'best_loss': float(self.best_loss)
                }
            }, f, default_flow_style=False)
    
    @staticmethod
    def load_hyperparameters(filepath: str) -> Hyperparameters:
        """Load hyperparameters from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return Hyperparameters(**data['hyperparameters'])


class Predictor:
    """Handles model prediction."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_loader = DataLoader(config)
    
    def predict(self, model_path: str, input_dir: Path, output_dir: Path):
        """Run predictions on files and copy high-confidence results."""
        model = tf.keras.models.load_model(model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths, data = self.data_loader.load_prediction_files(input_dir)
        
        for file_path, sample in zip(file_paths, data):
            sample_expanded = np.expand_dims(sample, axis=0)
            result = model.predict(sample_expanded, batch_size=1, steps=1)[0][0]
            
            if result > self.config.predict_threshold:
                print(f"{file_path.name}: {result:.4f}")
                shutil.copyfile(file_path, output_dir / file_path.name)


def load_config(config_path: str) -> Tuple[ModelConfig, Optional[Hyperparameters]]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Load model config
    model_data = data['model']
    training_data = data['training']
    prediction_data = data['prediction']
    weights_data = data['weights']
    
    config = ModelConfig(
        model_name=model_data['name'],
        max_length=model_data['max_length'],
        pad_value=model_data['pad_value'],
        max_params=model_data['max_params'],
        batch_size=training_data['batch_size'],
        epochs=training_data['epochs'],
        validation_split=training_data['validation_split'],
        patience=training_data['patience'],
        mode=training_data['mode'],
        predict_threshold=prediction_data['threshold'],
        positive_sample_weight=weights_data['positive_sample_weight'],
        positive_class_weight=weights_data['positive_class_weight']
    )
    
    # Load hyperparameters if present
    hp = None
    if 'hyperparameters' in data:
        hp = Hyperparameters(**data['hyperparameters'])
    
    return config, hp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train or run predictions with a binary file classifier using genetic algorithm optimization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  python script.py --config config.yml --mode train
  
  # Predict with config file
  python script.py --config config.yml --mode predict
  
  # Override model name
  python script.py --config config.yml --model-name my_model
  
  # Use custom data directories
  python script.py --config config.yml --positive-dir data/positive --negative-dir data/negative
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['train', 'predict'],
        help='Mode: train or predict (overrides config file)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name (overrides config file)'
    )
    
    parser.add_argument(
        '--positive-dir',
        type=str,
        help='Directory containing positive examples (class 1)'
    )
    
    parser.add_argument(
        '--negative-dir',
        type=str,
        help='Directory containing negative examples (class 0)'
    )
    
    parser.add_argument(
        '--predict-dir',
        type=str,
        help='Directory containing files to predict on'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for prediction results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config file)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config file)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config, initial_hp = load_config(args.config)
    
    # Apply command line overrides
    if args.mode:
        config.mode = args.mode
    if args.model_name:
        config.model_name = args.model_name
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Determine directories
    positive_dir = Path(args.positive_dir) if args.positive_dir else Path(config.model_name) / '1'
    negative_dir = Path(args.negative_dir) if args.negative_dir else Path(config.model_name) / '0'
    predict_dir = Path(args.predict_dir) if args.predict_dir else Path(config.model_name) / '0'
    output_dir = Path(args.output_dir) if args.output_dir else Path.home() / 'sscript'
    
    if config.mode == 'predict':
        # Prediction mode
        print(f"Running prediction mode")
        print(f"Model: {config.model_name}.h5")
        print(f"Input directory: {predict_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Threshold: {config.predict_threshold}")
        
        predictor = Predictor(config)
        predictor.predict(
            f'{config.model_name}.h5',
            predict_dir,
            output_dir
        )
    else:
        # Training mode
        print(f"Running training mode")
        print(f"Model: {config.model_name}")
        print(f"Positive examples: {positive_dir}")
        print(f"Negative examples: {negative_dir}")
        
        trainer = Trainer(config)
        data_loader = DataLoader(config)
        
        # Load data
        x_train, y_train, sample_weights = data_loader.load_dataset(
            positive_dir,
            negative_dir
        )
        
        print(f"Loaded {len(x_train)} samples")
        print(f"Positive samples: {np.sum(y_train)}")
        print(f"Negative samples: {len(y_train) - np.sum(y_train)}")
        
        # Load best hyperparameters if available
        hp_file = f'{config.model_name}_best_hp.yml'
        if Path(hp_file).exists():
            print(f"Loading existing hyperparameters from {hp_file}")
            initial_hp = Trainer.load_hyperparameters(hp_file)
        
        if initial_hp is None:
            raise ValueError(
                "No hyperparameters found. Please provide 'hyperparameters' section in config.yml"
            )
        
        print("Starting training with genetic algorithm optimization...")
        # Train
        trainer.train(x_train, y_train, sample_weights, initial_hp)


if __name__ == '__main__':
    main()