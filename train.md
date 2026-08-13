# Training the Local Scanner

Train the local detection model for the GPT Virus Scanner. The model learns to identify dangerous files by analyzing examples of safe and malicious code.

## Features

- **Simple settings:** Configure the training process with easy-to-read YAML files.
- **Smart optimization:** Find the best neural network structure automatically.
- **Easy execution:** Run training or prediction from your terminal with simple commands.
- **Broad support:** Analyze many different types of script files.
- **Automatic saving:** The tool saves your model and settings automatically as it improves.
- **Flexible size handling:** Process files of any size.

## Installation

1. **Install Python:** Ensure you have **Python 3.9, 3.10, 3.11, or 3.12** installed on your system.
2. **Create and activate a virtual environment (Recommended):**
   A virtual environment keeps your packages organized and avoids installation errors.

   - **macOS and Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (Command Prompt):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

   *Note: Activate the virtual environment whenever you open a new terminal. Run `deactivate` to exit it.*

<<<<<<< HEAD
3.  **Install requirements:** Open your terminal and run the appropriate command based on your operating system:

    *   **macOS and Linux:**
        *   **For Python 3.9, 3.10, or 3.11:**
            ```bash
            python3 -m pip install "tensorflow<2.16" numpy pyyaml
            ```
        *   **For Python 3.12:**
            ```bash
            python3 -m pip install tensorflow numpy pyyaml
            ```
    *   **Windows:**
        *   **For Python 3.9, 3.10, or 3.11:**
            ```cmd
            python -m pip install "tensorflow<2.16" numpy pyyaml
            ```
        *   **For Python 3.12:**
            ```cmd
            python -m pip install tensorflow numpy pyyaml
            ```
=======
3. **Install requirements:** Run the appropriate command for your Python version:
   - **For Python 3.9, 3.10, or 3.11:**
     ```bash
     python3 -m pip install "tensorflow<2.16" numpy pyyaml
     ```
   - **For Python 3.12:**
     ```bash
     python3 -m pip install tensorflow numpy pyyaml
     ```
>>>>>>> origin/docs-improve-train-guide-6270250563514395944

## Configuration

The trainer needs a configuration file (like `config.yml`). This file defines settings for the model, training parameters, and optimization choices.

### Example `config.yml`

Create a file named `config.yml` in your project folder and add the following settings:

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

# Optimization settings for the local model.
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

By default, the script organizes files inside a folder named after your model (defined by `name` under `model` in `config.yml`, which is `scripts` by default).

### Default Folder Layout

Place your training data in the following structure:

```
project/
├── config.yml
├── train.py
└── scripts/           # Matches the model name in config.yml
    ├── 0/             # Safe training files (Negative samples)
    │   ├── safe_script1.py
    │   └── safe_script2.js
    └── 1/             # Dangerous training files (Positive samples)
        ├── bad_script1.py
        └── bad_script2.sh
```

### Default Folders and Fallbacks

If you do not specify folder paths using command-line arguments, the tool uses these default folders automatically:

*   **Dangerous files folder (`--positive-dir`):** Defaults to `{model_name}/1/` (e.g., `scripts/1/`).
*   **Safe files folder (`--negative-dir`):** Defaults to `{model_name}/0/` (e.g., `scripts/0/`).
*   **Scan input folder (`--predict-dir`):** Defaults to `{model_name}/0/` (e.g., `scripts/0/`).
*   **Scan output folder (`--output-dir`):** Defaults to `~/sscript/` (a folder named `sscript` inside your user home directory). Any files flagged as suspicious during prediction are copied here.

---

## Usage

<<<<<<< HEAD
*Note: If you are on Windows, use `python` instead of `python3` for all the terminal examples shown below.*

### Basic Training
=======
### Training Mode
>>>>>>> origin/docs-improve-train-guide-6270250563514395944

To train a new detection model:

```bash
python3 train.py --config config.yml --mode train
```

*Note: The trainer automatically loads your best optimization settings from `{model_name}_best_hp.yml` if the file exists.*

### Prediction Mode (Scanning Files)

To scan files and find suspicious code using your trained model:

```bash
python3 train.py --config config.yml --mode predict
```

*Warning: Because no custom directories are specified, this command will scan your safe training files (`scripts/0/`) by default and copy any flagged files to your home directory (`~/sscript/`).*

#### Scan a Custom Folder

To scan a specific folder on your computer and save flagged files to a different output directory, use the `--predict-dir` and `--output-dir` options:

```bash
python3 train.py --config config.yml --mode predict --predict-dir /path/to/my_code --output-dir /path/to/flagged_results
```

---

## Advanced Options

You can override settings in `config.yml` directly from your command line:

```bash
# Use a custom model name (note: gptscan.py expects "scripts.h5")
python3 train.py --config config.yml --model-name custom_model

# Change training epochs and batch size
python3 train.py --config config.yml --epochs 50 --batch-size 64

# Use custom directories for safe and dangerous training files
python3 train.py --config config.yml --positive-dir data/malware --negative-dir data/safe
```

## Command-Line Options Reference

Use these flags to customize training and prediction:

| Flag | Description |
| :--- | :--- |
| `--config`, `-c` | **Required.** Path to your YAML configuration file (e.g., `config.yml`). |
| `--mode`, `-m` | Mode of operation: `train` or `predict`. |
| `--model-name` | Set a custom name for the model (influences saved file names and default paths). |
| `--positive-dir` | Folder containing dangerous training files. |
| `--negative-dir` | Folder containing safe training files. |
| `--predict-dir` | Folder containing files to scan during prediction. |
| `--output-dir` | Folder where flagged suspicious files are copied. |
| `--epochs` | Number of training rounds. |
| `--batch-size` | Number of files processed at once. |

---

## Output Files

When running, the tool creates these files:

- `{model_name}.h5` - The trained detection model. Rename this file to `scripts.h5` and place it in the project root folder to use it with `gptscan.py`.
- `{model_name}_best_hp.yml` - The best optimization settings found during the training process.

---

## How It Works

### The Training Process

1. **Gather files:** The trainer reads example files from your safe and dangerous folders.
2. **Standardize size:** The trainer converts each file into a standardized byte sequence:
   - For small files, it adds padding bytes (e.g., value `13`) up to the maximum length.
   - For large files, it extracts bytes from the beginning and the end to fit the limit.
3. **Optimize hyperparameters:** The trainer starts with your configuration settings and tests them.
4. **Learn and adjust:** The trainer continuously changes settings (two random parameters at a time) to find what works best:
   - If a new combination is more accurate, it saves it as the new best model.
   - It updates `{model_name}_best_hp.yml` with the improved settings.
5. **Continuous training:** This process runs infinitely. Press `Ctrl+C` to stop at any time.

### The Prediction Process

1. **Load model:** The predictor loads your trained `{model_name}.h5` file.
2. **Read files:** It reads all files from your input folder and standardizes their size.
3. **Calculate scores:** The model evaluates each file and assigns a threat score between `0.0` (safe) and `1.0` (dangerous).
4. **Filter results:** Files with threat scores above the threshold (e.g., `0.5`) are copied to your output folder for manual review.

---

## Tips

- Start with the default configuration settings.
- Let the training run for several hours to allow the optimization process to find the best model.
- Monitor your terminal for `New Best Model!` messages.
- Increase `positive_sample_weight` in `config.yml` if you have far more safe files than dangerous ones.
- Increase `max_params` if you want to allow larger, more complex models (this makes training slower).

## Stopping Training

Press `Ctrl+C` in your terminal to stop training. The trainer saves your progress and best configurations automatically, allowing you to resume training or run predictions at any time.
