# Training the Local Scanner

This tool trains the local "brain" (the file classifier) used by the GPT Virus Scanner. It learns to recognize the difference between safe and malicious files by studying many examples.

## Features

- **Simple settings:** Uses easy-to-read YAML files for configuration.
- **Smart optimization:** Automatically tries different settings to find the best way to detect threats.
- **Easy to use:** Run everything from your terminal with simple commands.
- **Broad support:** Analyzes many different types of files.
- **Automatic saving:** Keeps your progress safe by saving the best models as it finds them.
- **Flexible:** Handles files of any size automatically.

## Installation

1.  **Install Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet because of model compatibility.
2.  **Install requirements:** Open your terminal and run:
    ```bash
    pip install "tensorflow<2.16" numpy pyyaml
    ```

## Configuration

The trainer requires a `config.yml` file to run. This file contains settings for the model, the training process, and the AI settings.

### Example `config.yml`

Create a file named `config.yml` in your project folder and paste the following:

```yaml
# Settings for the local scanner trainer
model:
  name: "scripts"           # Name used for saved model files (e.g., scripts.h5)
  max_length: 1024          # Number of bytes analyzed from each file
  pad_value: 13             # Byte value used for padding small files
  max_params: 1000000       # Maximum allowed parameters (brain size)

training:
  batch_size: 32            # Files processed at once
  epochs: 100               # Number of training rounds
  validation_split: 0.2     # Percentage of data used for testing accuracy
  patience: 10              # Rounds to wait before stopping if no improvement
  mode: "train"             # Default mode: 'train' or 'predict'

prediction:
  threshold: 0.5            # Threat score (0.0 to 1.0) required to flag a file

weights:
  positive_sample_weight: 1.0 # Importance of malicious examples during training
  positive_class_weight: 1.0  # Weight applied to malicious files

# AI settings for the automatic optimization process.
# These values (0.0 to 1.0) control how the brain is built.
# The script will automatically adjust and improve these over time.
hyperparameters:
  embedding_scale: 0.5        # Size of the memory for byte patterns
  rnn_scale: 0.5              # Memory capacity for long sequences
  pooling_type: 0.5           # How patterns are summarized
  dropout1: 0.2               # Prevents the brain from memorizing specific files
  dense_scale: 0.5            # Complexity of the final decision layer
  activation: 0.1             # Mathematical style of the neurons
  dropout2: 0.2               # Additional prevention of over-memorization
  spatial_dropout: 0.1        # Pattern-based memorization prevention
  rnn_type: 0.1               # Type of memory layers used (LSTM or GRU)
  use_conv: 0.6               # Whether to use "vision" layers for patterns
  conv_filters_scale: 0.5     # Number of "vision" patterns to look for
  conv_padding: 0.1           # How patterns at the edges are handled
  kernel_init: 0.1            # Starting state of the brain's connections
  rnn_dropout: 0.1            # Reliability of memory connections
  rnn_recurrent_dropout: 0.1  # Reliability of internal memory feedback
  conv_kernel_scale: 0.5      # Size of the patterns to look for
  optimizer: 0.5              # How the brain learns from its mistakes
```

## Directory Structure

The script expects the following directory structure by default:

```
project/
├── config.yml
├── train.py
└── scripts/           # model_name from config
    ├── 0/             # safe files
    │   ├── file1.bin
    │   └── file2.bin
    └── 1/             # malicious files
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

**Continue training from the best AI settings:**
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
--positive-dir         Directory with malicious files
--negative-dir         Directory with safe files
--predict-dir          Directory with files to predict on
--output-dir           Output directory for prediction results
--epochs               Number of training rounds (overrides config)
--batch-size           Number of files processed at once (overrides config)
```

## Output Files

**Training mode produces:**
- `{model_name}.h5` - The trained "brain" (detection model).
- `{model_name}_best_hp.yml` - The best settings found during training.

**Prediction mode produces:**
- Copies of suspicious files (those with high threat scores) to your output folder.
- Terminal output showing each filename and its threat score.

## How It Works

### Training Process

1. **Gathers Examples:** The script looks at your folders of "safe" and "malicious" files.
2. **Prepares Data:** It converts the files into a standard format.
   - If a file is too small, it adds extra data to reach the required size.
   - If a file is too large, it takes parts from the beginning and the end.
3. **Tests Settings:** It starts with the settings you provided.
4. **Improves Automatically:** The script constantly tries new combinations of settings.
   - It changes two random settings at a time to see if the results get better.
   - If a new combination is more accurate, it becomes the new standard.
   - It saves the best version of the model automatically.
5. **Continuous Learning:** The process continues until you stop it manually (by pressing `Ctrl+C`).

### Prediction Process

1. **Loads the Brain:** The script loads your trained model.
2. **Scans New Files:** It looks at all files in your input folder.
3. **Assigns Scores:** It calculates how likely each file is to be malicious.
4. **Filters Results:** Any file that crosses your "threat" threshold is copied to the output folder for you to review.

## Automatic Setting Optimization

The script automatically tries different ways to build and train the model:

- **Structure:** How the "brain" is organized and how much it can remember.
- **Learning Style:** How it learns from its mistakes and how it processes information.
- **Special Layers:** Optional parts that can help it see patterns in the data more clearly.
- **Training Method:** The specific mathematical approaches used to improve the model's accuracy.

The script uses numbers between 0 and 1 to represent these settings. You can find the full list of how these numbers are used in `config.yml`.

## Tips

- Start with the default AI settings.
- Let training run for several hours to find the best settings.
- Watch the terminal for "New Best Model!" messages.
- Use the saved `*_best_hp.yml` file to resume training from the best settings.
- Adjust `positive_sample_weight` if you have many more safe files than malicious ones
- Increase `max_params` if you want larger models (though this will make training slower)

## Stopping Training

Press `Ctrl+C` to stop training. The best model and AI settings are already saved, so you can resume at any time.
