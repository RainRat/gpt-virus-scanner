# GPT Virus Scanner

## What is this?

GPT Virus Scanner uses AI to find malicious code in script files.

*   **Local Scan:** A fast, built-in model checks files on your computer.
*   **AI Analysis:** If a file looks suspicious, the tool can send it to an AI service (like OpenAI) for a detailed report.

**Note:** This tool is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, and PowerShell), Jupyter Notebooks (.ipynb), and archives (.zip, .tar), but does not analyze compiled programs.

## Quick Start

### Use the Windowed App (GUI)
1. Run the script: `python gptscan.py`
2. Select a folder to scan. The tool will also scan all subfolders.
3. Click **Scan now**.

### Use the Terminal (CLI)
Scan a folder and save a JSON report:
```bash
python gptscan.py ./my_scripts --cli -o report.json
```

Scan a code snippet from terminal input (piped):
```bash
echo "print('hello')" | python gptscan.py --cli --stdin
```

## Installation

### Prerequisites
*   **Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet because of model compatibility.
*   **Tkinter (Linux only):** If you are on Linux, you may need to install the Tkinter library (for example: `sudo apt-get install python3-tk`).

### Installation Steps
1.  **Download the code:** Clone this repository or download the zip file.
2.  **Check for required files:** Ensure `gptscan.py`, `scripts.h5`, `task.txt`, and `extensions.txt` are all in the same folder.
    *   **Note:** Always run the scanner from inside its own folder so it can find these files.
3.  **Install dependencies:** Open your terminal and run:
    ```bash
    pip install "tensorflow<2.16" openai
    ```

## AI Service Setup

To use the **AI Analysis** feature, you must set up an AI provider and then enable the feature in the scanner. Your code is only sent to an AI service if you choose to enable this option.

### Step 1: Set up a Provider

#### Cloud-based (OpenAI or OpenRouter)
These services process your code in the cloud and require an API key.
1.  **Get an API key:** Sign up for [OpenAI](https://openai.com/) or [OpenRouter](https://openrouter.ai/).
2.  **Add your key:** You have two options:
    *   Create a file named `apikey.txt` in the scanner folder and paste your key on the first line.
    *   Set the `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable in your terminal.

#### Local AI (Ollama)
Ollama runs entirely on your own computer. It is private and **does not require an API key**.
1.  **Install Ollama:** Download and install [Ollama](https://ollama.com/).
2.  **Download a model:** Run `ollama pull llama3.2` (or your preferred model) in your terminal.
3.  **Run Ollama:** Ensure the Ollama app is running before you start the scanner.

### Step 2: Enable AI Analysis

Once your provider is ready, you must enable the feature when you run a scan:
*   **In the App Window:** Check the **Use AI Analysis** box before clicking **Scan now**.
*   **In the Terminal:** Use the `--use-gpt` flag (for example: `python gptscan.py ./my_scripts --cli --use-gpt`).

## Supported Files

The scanner finds scripts in four ways:
*   **By file type:** It recognizes common script types (like `.py`, `.js`, `.sh`, and `.ps1`) and Jupyter Notebooks (`.ipynb`) using the included `extensions.txt` file.
*   **By archive content:** It can inspect scripts hidden inside `.zip`, `.tar`, and `.tar.gz` files.
*   **By the first line of the file:** If a file does not have an extension, the tool checks the very first line to identify the script type (for example, a line starting with `#!/bin/bash`).
*   **Remote scripts (via URL):** It can download and scan scripts directly from the web using HTTP or HTTPS links.

## Configuration

You can customize the scanner using these files in the same folder:
*   `apikey.txt`: Your AI service API key.
*   `extensions.txt`: A list of file extensions to scan (one per line).
*   `.gptscanignore`: Patterns of files or folders to skip (one per line). For example:
    ```text
    node_modules/*
    *.log
    temp_dir/
    ```
*   `task.txt`: Instructions for the AI analysis.

## How to Use

### Using the App Window

Run `python gptscan.py` to open the GUI.

*   **Select File/Folder:** Choose what you want to scan. If you select a folder, the tool scans all files inside it and its subfolders. The path input is a dropdown that remembers your last 10 scan locations.
*   **URL:** Scan a script or archive (.zip, .tar, .tar.gz) directly from a web link.
*   **Clipboard:** Scan code currently in your clipboard.
*   **Filter results:** Search findings by path, confidence, notes, or code snippets.
*   **Deep Scan:** Check the entire file. By default, the scanner only checks the first and last 1024 bytes to save time.
*   **Scan all files:** Scan all files regardless of their extension or whether they contain a script starting line (like #!/bin/bash).
*   **Minimum Threat Level:** Set the sensitivity. Higher values show only the most dangerous files.
*   **Show all files:** See every scanned file, even safe ones.
*   **Use AI Analysis:** Enable detailed reports for suspicious findings.
*   **Batch AI Analysis:** Select multiple findings and analyze them all at once using the "Analyze with AI" button or right-click menu.
*   **VirusTotal:** Look up a file's signature on VirusTotal to see other security reports.
*   **Open / Reveal:** Open the selected file or show its location in your folder.
*   **Manage Exclusions:** View and edit your `.gptscanignore` file to skip specific patterns.
*   **Copy CLI Command:** Copy the command-line equivalent of your current settings.
*   **Import/Export:** Save or load results.
    *   **Import:** Supports CSV, JSON, JSONL, NDJSON, and SARIF formats. You can also import results directly from your clipboard.
    *   **Export:** Supports CSV, JSON, HTML, SARIF, and Markdown formats.

**Shortcuts:**
*   **Ctrl+A / Cmd+A:** Select all results.
*   **Ctrl+F / Cmd+F:** Focus the search filter.
*   **Ctrl+C / Cmd+C:** Copy path(s) of selected result(s).
*   **Ctrl+Shift+C / Cmd+Shift+C:** Copy selected result(s) as a Markdown table.
*   **Ctrl+G / Cmd+G:** Analyze selected result(s) with AI.
*   **Ctrl+H / Cmd+H:** Copy SHA256 hash(es) of selected result(s).
*   **Ctrl+J / Cmd+J:** Copy selected result(s) as JSON.
*   **Ctrl+S / Cmd+S:** Copy code snippet(s) of selected result(s).
*   **Ctrl+V / Cmd+V:** Import results from clipboard.
*   **Ctrl+Return / Cmd+Return:** Reveal selected file in folder.
*   **F5 / R:** Rescan selected files.
*   **Double-click / Enter / Space:** View detailed analysis and code.
*   **Shift+Enter:** Open selected file.
*   **Ctrl+Shift+E / Cmd+Shift+E:** Copy the current scan settings as a CLI command.
*   **Delete:** Exclude selected results from future scans.
*   **Esc:** Cancel the active scan.

### Using the Command Line (CLI)

Run scans from your terminal using the `--cli` flag.

**Examples:**
```bash
# Basic scan with AI analysis
python gptscan.py ./my_scripts --cli --use-gpt

# Scan a code snippet from terminal input (piped)
echo "print('hello')" | python gptscan.py --cli --stdin

# Scan using Ollama (local AI)
python gptscan.py ./my_scripts --cli --use-gpt --provider ollama --model llama3.2

# Save results to a JSON file
python gptscan.py ./my_scripts --cli -o results.json --exclude "tests/*"

# Scan a remote script via URL
python gptscan.py https://example.com/script.sh --cli

# Convert an existing JSON report to an HTML report
python gptscan.py --cli --import results.json -o report.html
```

**Common Options:**
*   `--cli`: Run in command-line mode.
*   `--stdin`: Read a code snippet from terminal input (piped) to scan.
*   `--deep`: Scan the entire file.
*   `--dry-run`: Show which files would be scanned without analyzing them.
*   `--show-all`: List all files, even safe ones.
*   `--use-gpt`: Enable AI Analysis for suspicious code.
*   `--output [file], -o [file]`: Save results to a file. The format is chosen based on the extension (.json, .csv, .html, .sarif, .md).
*   `--threshold [0-100], -t [0-100]`: The lowest threat score to report (default: 50).
*   `--fail-threshold [0-100]`: Exit with an error if any file meets this threat level.
*   `--git-changes`: Only scan files that have changed in Git.
*   `--all-files`: Scan all files regardless of their extension or whether they contain a script starting line (like #!/bin/bash).
*   `--exclude [patterns], -e [patterns]`: Skip files matching these patterns.
*   `--extensions [types]`: Only scan specific file types (for example: `py,js`).
*   `--import [file]`: Load results from a previous scan (JSON, CSV, or SARIF). Use `-` to read from terminal input (piped).
*   `--markdown`: Save the report in Markdown format.

## Advanced: Training

You can retrain the local scanner model to recognize new types of threats. For detailed instructions on how to prepare your data and run the trainer, see the [Training Guide](train.md).

## Troubleshooting

*   **Tkinter not found:** On Linux, run `sudo apt-get install python3-tk`.
*   **Model file missing:** Ensure `scripts.h5` is in the same folder as `gptscan.py`. This file is required for the scanner to function.
*   **Extensions list missing:** Ensure `extensions.txt` exists in the same folder. This file is required to detect script files by their extension.
*   **AI Analysis disabled:** Ensure `task.txt` exists in the same folder. Detailed AI reports will not work without it.
*   **AI Analysis results not showing:** Ensure you have checked the **Use AI Analysis** box (GUI) or added the `--use-gpt` flag (CLI). If you are using OpenAI or OpenRouter, double-check that your API key is correct in `apikey.txt` or your environment variables.

## Contributing

We welcome your help! Please run tests before submitting a Pull Request:
```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov
PYTHONPATH=. python3 -m pytest
```

## License

LGPL 2.1 or later
