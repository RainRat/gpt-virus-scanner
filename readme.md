# GPT Virus Scanner

## What is this?

GPT Virus Scanner uses AI to find malicious code in script files. It works in two stages:
1.  **Local Scan:** A fast, built-in model checks your files on your computer.
2.  **AI Analysis:** If a file looks suspicious, the tool sends it to an AI service (like OpenAI) for a detailed report.

**Note:** This tool is a prototype. It is not a commercial antivirus product. It scans scripts (such as Python, JavaScript, PowerShell, and Batch files) but does not analyze compiled programs or compressed files (like .zip).

## Quick Start

### Use the Windowed App (GUI)
1. Start the scanner: `python gptscan.py`
2. Choose a folder to scan.
3. Click **Scan now**.

### Use the Terminal (CLI)
Scan a folder and generate a JSON report:
```bash
python gptscan.py ./my_scripts --cli --json
```

## Requirements

*   **Python 3.9, 3.10, or 3.11** is required. Newer versions (like 3.12) are currently not compatible with the local scanner.
*   **TensorFlow** (version 2.15 or older) for local scanning.
*   **AI Service** (Optional, for detailed analysis):
    *   **OpenAI** or **OpenRouter** (requires an API key).
    *   **Ollama** (runs locally; no API key needed).
*   **Tkinter** for the graphical interface.

## Installation

1.  **Download the tool:** Clone this repository or download the files. Make sure `gptscan.py` and `scripts.h5` are in the same folder.
2.  **Install Python:** Download and install Python (version 3.9, 3.10, or 3.11) from [python.org](https://www.python.org/).
3.  **Install libraries:** Open your terminal and run:
    ```bash
    pip install "tensorflow<2.16" openai
    ```
    *Note for Linux users:* You may need to install Tkinter using your package manager (for example, `sudo apt-get install python3-tk` on Ubuntu).
4.  **Set up AI (Optional):** To use AI Analysis, make sure `task.txt` is in the same folder and set up an API key.

    If you want to use **AI Analysis** (OpenAI or OpenRouter), you need an API key:
    *   Create a file named `apikey.txt` in the same folder as `gptscan.py`.
    *   Paste your API key into that file. It should contain only the key on a single line.
    *   **Alternatively:** Use the `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable.

    *   **OpenAI:** Get a key from [OpenAI](https://platform.openai.com/).
    *   **OpenRouter:** Get a key from [OpenRouter](https://openrouter.ai/).
    *   **Ollama:** Download [Ollama](https://ollama.com/) and ensure it is running. You must also download a model to use it (for example, run `ollama pull llama3.2` in your terminal).

    *Privacy Note:* Your code is only sent to the AI service if you enable the "Use AI Analysis" option. Check your provider's data policy.

## Supported Files

The scanner finds scripts to analyze in two ways:
*   **By extension:** It looks for `.py`, `.js`, `.bat`, and `.ps1` files by default. The `extensions.txt` file helps the scanner recognize many other file types, such as `.sh`, `.rb`, and `.php`.
*   **By "shebang" line:** The tool identifies scripts without extensions by looking at the first line (for example, `#!/bin/bash`). It supports common interpreters like Python, Node.js, Bash, Sh, Zsh, PHP, Perl, Ruby, and PowerShell.

## Configuration

You can customize the scanner using these optional files in the same folder:

*   **`apikey.txt`**: Your API key for OpenAI or OpenRouter.
*   **`extensions.txt`**: A list of file extensions to scan (e.g., `.py`, one per line).
*   **`.gptscanignore`**: Patterns of files or folders to skip (like a `.gitignore` file).
*   **`task.txt`**: The instructions given to the AI for its analysis.

## How to Use

### Using the App Window

Run the script to open the window:
```bash
python gptscan.py
```

*   **Select File/Folder:** Choose the specific script or folder you want to scan.
*   **Filter results:** Quickly search through findings by filename, confidence level, or analysis notes.
*   **Deep Scan:** Scan every part of the file. By default, the tool only checks the beginning and end (the first and last 1,024 bytes) to save time.
*   **Minimum Threat Level:** Set the threshold (0-100%) for what the tool considers "suspicious."
*   **Show all files:** List every file scanned, even the safe ones.
*   **Use AI Analysis:** Get a detailed report for suspicious files.
*   **AI Analysis Settings:** Choose your service (OpenAI, OpenRouter, or Ollama) and the model you want to use.
*   **Import/Export Results:** Save your results to CSV, JSON, HTML, SARIF, or Markdown files. You can load CSV, JSON, and SARIF files back into the tool later. (Markdown is for export only).

You can sort the results by clicking the column headers.

**Shortcuts:**
*   **Ctrl+A / Cmd+A:** Select all results.
*   **F5 / R:** Rescan selected files.
*   **Double-click / Enter:** Open selected file.
*   **Space:** View detailed analysis and code for the selected result.
*   **Esc:** Cancel the active scan.

![Scan Results](gpt-virus-scan.png)

### Using the Command Line (CLI)

Run scans from your terminal for automated tasks. Use the `--cli` flag.

> **Tip:** By default, CLI results appear in your terminal. To save them to a file, use the `--output` (or `-o`) flag or shell redirection (`>`). For example: `python gptscan.py ./my_scripts --cli -o results.json`.

**Examples:**
```bash
# Basic scan with AI analysis
python gptscan.py ./my_scripts --cli --use-gpt

# Scan using Ollama (local AI)
python gptscan.py ./my_scripts --cli --use-gpt --provider ollama --model llama3.2

# Export results to a JSON file
python gptscan.py ./my_scripts --cli --json --exclude "tests/*" > results.json
```

**Common Options:**
*   `--cli`: Run in command-line mode (required for terminal use).
*   `[target]`: The folder or file to scan.
*   `--path [path]`: An alternative way to specify the scan target.
*   `--deep`: Scan the entire file instead of just the beginning and end.
*   `--show-all`: List all files, even safe ones.
*   `--use-gpt`: Enable AI Analysis for suspicious code.
*   `--output [file], -o [file]`: Save scan results to a file. The format is inferred from the extension (.json, .csv, .html, .sarif, .md).
*   `--json`: Output results in JSON format (one object per line).
*   `--csv`: Output results in CSV format (default).
*   `--sarif`: Save results in SARIF format (standard for security tools).
*   `--html`: Create a standalone HTML report.
*   `--md, --markdown`: Create a Markdown report of the results.
*   `--dry-run`: List files that would be scanned without analyzing them.
*   `--extensions "py,js,bat"`: Scan specific file types (separated by commas).
*   `--exclude [patterns]`: Skip files matching these patterns (e.g., `node_modules/*`).
*   `--file-list [file]`: Read a list of files to scan from a text file.
*   `--git-changes`: Only scan files that have changed in your git repository.
*   `--provider [name]`: Choose your AI service ('openai', 'openrouter', or 'ollama').
*   `--model [name]`: Specify the exact AI model (e.g., 'gpt-4o', 'llama3.2').
*   `--api-base [url]`: Set a custom URL for the AI service (useful for local servers).
*   `--rate-limit [n]`: Limit AI requests per minute to avoid errors (default: 60).
*   `--fail-threshold [0-100]`: Return a failure code if any file meets this threat level.
*   `--version`: Show the tool's version.

## Troubleshooting

*   **Tkinter not found (Linux):** If you see a "ModuleNotFoundError: No module named 'tkinter'" error, install it using: `sudo apt-get install python3-tk`.
*   **Model file missing:** Make sure `scripts.h5` is in the same folder as `gptscan.py`. This file is required for the local scan.
*   **AI Analysis disabled:** Ensure `task.txt` exists in the same folder. This file contains the instructions for the AI.
*   **API key missing:** You can still use the local scan and Ollama without a key. For OpenAI or OpenRouter, add your key to `apikey.txt`.

## Contributing

We welcome your help!

*   **Reporting issues:** Let us know if the tool misses something or gives a false alarm.
*   **Submitting code:** Pull requests are welcome. Please run tests before submitting:
    ```bash
    pip install pytest pytest-asyncio pytest-mock pytest-cov
    python -m pytest --cov
    ```

## Credits

Thanks to the Stack Overflow community for GUI inspiration.

## License

LGPL 2.1 or later
