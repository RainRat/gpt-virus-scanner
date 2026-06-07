# Training the Local Scanner

Train the local model (the file classifier) for the GPT Virus Scanner. It learns to find dangerous files by studying many examples of safe and malicious code.

## Features

- **Simple settings:** Use easy-to-read YAML files for configuration.
- **Smart optimization:** Find the best way to detect threats automatically.
- **Easy to use:** Run everything from your terminal with simple commands.
- **Broad support:** Analyze many different types of files.
- **Automatic saving:** Keep your progress safe with automatic model saves.
- **Flexible:** Process files of any size automatically.

## Installation

1.  **Install Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet because of model compatibility.
2.  **Install requirements:** Open your terminal and run:
    ```bash
    python3 -m pip install "tensorflow<2.16" numpy pyyaml
    ```

## Configuration

The trainer requires a `config.yml` file. This file contains settings for the model, the training process, and the optimization settings.

### Example `config.yml`

Create a file named `config.yml` in your project folder and paste the following:

```yaml
# Settings for the local scanner trainer
model:
  name: "scripts"           # Name used for saved model files (e.g., scripts.h5)
  max_length: 1024          # Number of bytes analyzed from each file
  pad_value: 13             # Byte value used for padding small files
  max_params: 1000000       # Maximum allowed parameters (model size)

training:
  batch_size: 32            # Files processed at once
  epochs: 100               # Number of training rounds
  validation_split: 0.2     # Percentage of data used for testing accuracy
  patience: 10              # Rounds to wait before stopping if no improvement
  mode: "train"             # Default mode: 'train' or 'predict'

prediction:
  threshold: 0.5            # Threat level (0.0 to 1.0) required to flag a file

weights:
  positive_sample_weight: 1.0 # Importance of dangerous examples during training

# Optimization settings (Hyperparameters) for the local model.
# These values (0.0 to 1.0) control how the model is built.
# The script will automatically adjust and improve these over time.
hyperparameters:
  embedding_scale: 0.5        # Size of the memory for byte patterns
  rnn_scale: 0.5              # Memory capacity for long sequences
  pooling_type: 0.5           # How patterns are summarized
  dropout1: 0.2               # Prevents the model from memorizing specific files
  dense_scale: 0.5            # Complexity of the final decision layer
  activation: 0.1             # Mathematical style of the connections
  dropout2: 0.2               # Additional prevention of over-memorization
  spatial_dropout: 0.1        # Pattern-based memorization prevention
  rnn_type: 0.1               # Type of memory layers used (LSTM or GRU)
  use_conv: 0.6               # Whether to use "vision" layers for patterns
  conv_filters_scale: 0.5     # Number of "vision" patterns to look for
  conv_padding: 0.1           # How patterns at the edges are handled
  kernel_init: 0.1            # Starting state of the model's connections
  rnn_dropout: 0.1            # Reliability of memory connections
  rnn_recurrent_dropout: 0.1  # Reliability of internal memory feedback
  conv_kernel_scale: 0.5      # Size of the patterns to look for
  optimizer: 0.5              # How the model learns from its mistakes
```

## Folder Structure

The script expects the following folder structure by default:

```
project/
├── config.yml
├── train.py
└── scripts/           # model_name from config
    ├── 0/             # safe files
    │   ├── file1.bin
    │   └── file2.bin
    └── 1/             # dangerous files
        ├── file3.bin
        └── file4.bin
```

## Usage

### Basic Training

Train a model using the config file:

```bash
python3 train.py --config config.yml --mode train
```

### Basic Prediction

Run predictions on files:

```bash
python3 train.py --config config.yml --mode predict
```

### Advanced Options

#### Override settings:

```bash
# Use a different model name
python3 train.py --config config.yml --model-name my_model

# Change training parameters
python3 train.py --config config.yml --epochs 50 --batch-size 64

# Specify custom folders for training data
python3 train.py --config config.yml \
    --positive-dir /path/to/positive \
    --negative-dir /path/to/negative
```

#### Custom scan folders:

```bash
python3 train.py --config config.yml \
    --mode predict \
    --predict-dir /path/to/input \
    --output-dir /path/to/output
```

### Complete Examples

**Train a new model:**
```bash
python3 train.py --config config.yml --mode train
```

**Continue training from the best optimization settings:**
```bash
# The script automatically loads scripts_best_hp.yml if it exists
python3 train.py --config config.yml --mode train
```

**Predict with custom threshold:**
```bash
# Edit config.yml to set prediction.threshold: 0.9
python3 train.py --config config.yml --mode predict
```

**Train with different data layout:**
```bash
python3 train.py --config config.yml \
    --mode train \
    --positive-dir data/malware \
    --negative-dir data/benign \
    --model-name malware_detector
```

## Command-Line Options

```
--config, -c           Path to the YAML settings file (required).
--mode, -m             Choose between training a model or making predictions.
--model-name           Set the model name.
--positive-dir         Folder containing dangerous files.
--negative-dir         Folder containing safe files.
--predict-dir          Folder containing files to scan.
--output-dir           Folder where suspicious files will be copied.
--epochs               Number of training rounds.
--batch-size           Number of files to process at once.
```

## Output Files

**Training mode produces:**
- `{model_name}.h5` - The trained detection model.
- `{model_name}_best_hp.yml` - The best settings found during training.

**Prediction mode produces:**
- Copies of suspicious files (those with high threat levels) to your output folder.
- Terminal output showing each filename and its threat level.

## How It Works

### Training Process

1. **Gathers Examples:** The trainer scans your folders for "safe" and "dangerous" files.
2. **Prepares Data:** The trainer converts the files into a standard format.
   - If a file is too small, it adds extra data to reach the required size.
   - If a file is too large, the trainer takes pieces from the beginning and the end to fit the model's size limit.
3. **Tests Settings:** The trainer starts with the settings you provided.
4. **Improves Automatically:** The trainer constantly tries new combinations of settings.
   - It changes two random settings at a time to see if the results get better.
   - If a new combination is more accurate, it becomes the new standard.
   - It saves the best version of the model automatically.
5. **Continuous Learning:** The process continues until you stop it manually (by pressing `Ctrl+C`).

### Prediction Process

1. **Loads the Model:** The trainer loads your trained model.
2. **Scans New Files:** It scans all files in your input folder.
3. **Assigns Scores:** It calculates the threat level for each file.
4. **Filters Results:** Any file that crosses your "threat" threshold is copied to the output folder for you to review.

## Automatic Setting Optimization

The trainer automatically tries different ways to build and train the model:

- **Structure:** How the model is organized and how much it can remember.
- **Learning Style:** How it learns from its mistakes and how it processes information.
- **Special Layers:** Optional parts that can help it see patterns in the data more clearly.
- **Training Method:** The specific mathematical approaches used to improve the model's accuracy.

The trainer uses numbers between 0 and 1 to represent these settings. You can find the full list of how these numbers are used in `config.yml`.

## Tips

- Start with the default optimization settings.
- Let training run for several hours to find the best settings.
- Watch the terminal for "New Best Model!" messages.
- Use the saved `*_best_hp.yml` file to resume training from the best settings.
- Adjust `positive_sample_weight` if you have many more safe files than dangerous ones
- Increase `max_params` if you want larger models (though this will make training slower)

## Stopping Training

Press `Ctrl+C` to stop training. The trainer already saved the best model and optimization settings, so you can resume at any time.
