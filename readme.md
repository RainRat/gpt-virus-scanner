# GPT Virus Scanner

## What is this?

GPT Virus Scanner uses AI to find malicious code in script files.

*   **Local Scan:** A fast, built-in model checks files on your computer.
*   **AI Analysis:** If a file looks suspicious, the tool can send it to an AI service (like OpenAI) for a detailed report.

**Note:** This tool is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, and PowerShell) but does not analyze compiled programs or compressed files (like .zip).

## Quick Start

### Use the Windowed App (GUI)
1. Run the script: `python gptscan.py`
2. Select a folder to scan.
3. Click **Scan now**.

### Use the Terminal (CLI)
Scan a folder and save a JSON report:
```bash
python gptscan.py ./my_scripts --cli -o report.json
```

## Setup

Follow these steps to get the scanner running:

1.  **Download the code:** Clone this repository or download the zip file. Ensure `gptscan.py` and `scripts.h5` are in the same folder. The `scripts.h5` file is mandatory for scanning.
2.  **Install Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not yet supported because of model compatibility. You can download it from [python.org](https://www.python.org/).
3.  **Install dependencies:** Open your terminal and run:
    ```bash
    pip install "tensorflow<2.16" openai
    ```
    *Linux users:* You may also need to install Tkinter (e.g., `sudo apt-get install python3-tk`).
4.  **Set up AI (Optional):** To use detailed AI analysis, make sure `task.txt` is in the folder.
    *   **OpenAI/OpenRouter:** Create a file named `apikey.txt` and paste your API key inside. Alternatively, set the `OPENAI_API_KEY` environment variable.
    *   **Ollama:** Download [Ollama](https://ollama.com/) and run it locally. Pull a model before starting (e.g., `ollama pull llama3.2`).

*Privacy Note:* Your code is only sent to an AI service if you enable "Use AI Analysis."

## Supported Files

The scanner identifies scripts in two ways:
*   **By extension:** It looks for `.py`, `.js`, `.bat`, and `.ps1` files by default. It can recognize many others (like `.sh`, `.rb`, and `.php`) if they are listed in `extensions.txt`.
*   **By "shebang" line:** The tool can identify scripts without extensions by reading the first line (for example, `#!/bin/bash`).

## Configuration

You can customize the scanner using these files in the same folder:
*   `apikey.txt`: Your AI service API key.
*   `extensions.txt`: A list of file extensions to scan (one per line).
*   `.gptscanignore`: Patterns of files or folders to skip.
*   `task.txt`: Instructions for the AI analysis.

## How to Use

### Using the App Window

Run `python gptscan.py` to open the GUI.

*   **Select File/Folder:** Choose what you want to scan.
*   **Filter results:** Search findings by path, confidence, notes, or code snippets.
*   **Deep Scan:** Check the entire file. By default, it only checks the beginning and end to save time.
*   **Minimum Threat Level:** Set the sensitivity. Higher values show only the most dangerous files.
*   **Show all files:** See every scanned file, even safe ones.
*   **Use AI Analysis:** Enable detailed reports for suspicious findings.
*   **Import/Export:** Save or load results in CSV, JSON, HTML, or SARIF formats. (Markdown is for export only).

**Shortcuts:**
*   **Ctrl+A / Cmd+A:** Select all results.
*   **Ctrl+F / Cmd+F:** Focus the search filter.
*   **F5 / R:** Rescan selected files.
*   **Double-click / Enter / Space:** View detailed analysis and code.
*   **Shift+Enter:** Open selected file.
*   **Esc:** Cancel the active scan.

### Using the Command Line (CLI)

Run scans from your terminal using the `--cli` flag.

**Examples:**
```bash
# Basic scan with AI analysis
python gptscan.py ./my_scripts --cli --use-gpt

# Scan using Ollama (local AI)
python gptscan.py ./my_scripts --cli --use-gpt --provider ollama --model llama3.2

# Save results to a JSON file
python gptscan.py ./my_scripts --cli -o results.json --exclude "tests/*"
```

**Common Options:**
*   `--cli`: Run in command-line mode.
*   `--deep`: Scan the entire file.
*   `--show-all`: List all files, even safe ones.
*   `--use-gpt`: Enable AI Analysis for suspicious code.
*   `--output [file], -o [file]`: Save results to a file. The format is chosen based on the extension (.json, .csv, .html, .sarif, .md).
*   `--threshold [0-100], -t [0-100]`: The lowest threat score to report (default: 50).
*   `--fail-threshold [0-100]`: Exit with an error if any file meets this threat level.
*   `--git-changes`: Only scan files that have changed in Git.
*   `--exclude [patterns], -e [patterns]`: Skip files matching these patterns.

## Troubleshooting

*   **Tkinter not found:** On Linux, run `sudo apt-get install python3-tk`.
*   **Model file missing:** Ensure `scripts.h5` is in the same folder as `gptscan.py`. This file is required for the scanner to function.
*   **AI Analysis disabled:** Ensure `task.txt` exists in the same folder. Detailed AI reports will not work without it.

## Contributing

We welcome your help! Please run tests before submitting a Pull Request:
```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov
PYTHONPATH=. python3 -m pytest
```

## License

LGPL 2.1 or later
