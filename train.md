# Neural Network Binary Classifier

A refactored binary file classifier using genetic algorithm optimization for hyperparameter tuning.

## Features

- YAML-based configuration
- Genetic algorithm for hyperparameter optimization
- Command-line interface with flexible options
- Support for binary file classification
- Automatic model checkpointing
- Custom padding logic for variable-length files

## Installation

```bash
pip install tensorflow numpy pyyaml
```

## Configuration

## Directory Structure

The script expects the following directory structure by default:

```
project/
├── config.yml
├── train.py
└── scripts/           # model_name from config
    ├── 0/             # negative examples (class 0)
    │   ├── file1.bin
    │   └── file2.bin
    └── 1/             # positive examples (class 1)
        ├── file3.bin
        └── file4.bin
```

## Usage

### Basic Training

Train a model using the config file:

```bash
python train.py --config config.yml --mode train
```

### Basic Prediction

Run predictions on files:

```bash
python train.py --config config.yml --mode predict
```

### Advanced Options

#### Override configuration values:

```bash
# Use different model name
python train.py --config config.yml --model-name my_model

# Change training parameters
python train.py --config config.yml --epochs 50 --batch-size 64

# Specify custom data directories
python train.py --config config.yml \
    --positive-dir /path/to/positive \
    --negative-dir /path/to/negative
```

#### Custom prediction directories:

```bash
python train.py --config config.yml \
    --mode predict \
    --predict-dir /path/to/input \
    --output-dir /path/to/output
```

### Complete Examples

**Train a new model:**
```bash
python train.py --config config.yml --mode train
```

**Continue training from best hyperparameters:**
```bash
# The script automatically loads scripts_best_hp.yml if it exists
python train.py --config config.yml --mode train
```

**Predict with custom threshold:**
```bash
# Edit config.yml to set prediction.threshold: 0.9
python train.py --config config.yml --mode predict
```

**Train with different data layout:**
```bash
python train.py --config config.yml \
    --mode train \
    --positive-dir data/malware \
    --negative-dir data/benign \
    --model-name malware_detector
```

## Command-Line Options

```
--config, -c           Path to YAML configuration file (required)
--mode, -m             Mode: train or predict (overrides config)
--model-name           Model name (overrides config)
--positive-dir         Directory with positive examples (class 1)
--negative-dir         Directory with negative examples (class 0)
--predict-dir          Directory with files to predict on
--output-dir           Output directory for prediction results
--epochs               Number of training epochs (overrides config)
--batch-size           Batch size (overrides config)
```

## Output Files

**Training mode produces:**
- `{model_name}.h5` - Trained Keras model
- `{model_name}_best_hp.yml` - Best hyperparameters found

**Prediction mode produces:**
- Copies of high-confidence files to the output directory
- Console output showing filename and confidence score

## How It Works

### Training

1. Loads positive and negative examples from specified directories
2. Applies custom padding logic:
   - Files shorter than max_length are padded with pad_value (default: 13)
   - Files longer than max_length are split: first half + last half
3. Starts with initial hyperparameters from config
4. Uses genetic algorithm to optimize:
   - Mutates 2 random hyperparameters each iteration
   - Keeps best model based on validation loss
   - Saves improved models automatically
5. Runs indefinitely until manually stopped (Ctrl+C)

### Prediction

1. Loads trained model
2. Processes files from input directory
3. Predicts probability for each file
4. Copies files exceeding threshold to output directory

## Hyperparameter Optimization

The genetic algorithm mutates these parameters:

- **Architecture**: Embedding size, RNN type (LSTM/GRU/Bidirectional), pooling strategy
- **Regularization**: Dropout rates, spatial dropout
- **Convolution**: Optional Conv1D layer with varying filters and kernel sizes
- **Training**: Optimizer selection, kernel initializers

Each parameter is normalized to [0, 1] and mapped to specific values. See `config.yml` for detailed mappings.

## Tips

- Start with the provided default hyperparameters
- Let training run for several hours to explore the hyperparameter space
- Monitor the console output for "New Best Model!" messages
- Use the saved `*_best_hp.yml` file to resume training from the best configuration
- Adjust `positive_sample_weight` if you have class imbalance
- Increase `max_params` if you want larger models (at cost of training time)

## Stopping Training

Press `Ctrl+C` to stop training. The best model and hyperparameters are already saved, so you can resume anytime.