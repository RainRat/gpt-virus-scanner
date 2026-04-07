# GPT Virus Scanner

## What is this?

GPT Virus Scanner uses AI to find malicious code in script files.

*   **Local Scan:** A fast, built-in model checks files on your computer.
*   **AI Analysis:** If a file looks suspicious, the tool can send it to an AI service (like OpenAI) for a detailed report.

**Note:** This tool is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, and PowerShell), Jupyter Notebooks (.ipynb), Markdown files (.md), HTML files (.html, .htm), and archives (.zip, .tar), but does not analyze compiled programs.

## Quick Start

### Use the Windowed App (GUI)
1. Run the script: `python gptscan.py`
2. Select a folder to scan. By default, it opens the last used folder or the current directory.
3. Click **Scan Now**.

### Use the Terminal (CLI)
Scan a folder and save a JSON report:
```bash
python gptscan.py ./my_scripts --cli -o report.json
```

Scan multiple folders at once:
```bash
python gptscan.py ./folder1 ./folder2 --cli
```

Scan a code snippet sent from another command in the terminal:
```bash
echo "print('hello')" | python gptscan.py --cli --stdin
```

## Installation

### Prerequisites
*   **Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet due to model compatibility.
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
*   **In the App Window:** Check the **Use AI Analysis** box before clicking **Scan Now**.
*   **In the Terminal:** Use the `--use-gpt` flag (for example: `python gptscan.py ./my_scripts --cli --use-gpt`).

## Privacy and Security

We value your privacy and the security of your code.

*   **Local Scans:** When you run a standard scan, all analysis happens entirely on your own computer. No code is sent to any external service.
*   **AI Analysis (Optional):** If you enable AI Analysis, the tool only sends a small **1 KB snippet** of suspicious code to the AI provider—not the entire file. This helps protect your intellectual property and sensitive data.
*   **Ollama (Local AI):** If you use Ollama, all AI analysis stays on your computer. No data is sent to the cloud.

## Supported Files

The scanner finds scripts in several ways:
*   **By file type:** It recognizes common script types (like `.py`, `.js`, `.sh`, and `.ps1`) and Jupyter Notebooks (`.ipynb`) using the included `extensions.txt` file.
*   **By archive content:** It can inspect scripts hidden inside `.zip`, `.tar`, and `.tar.gz` files.
*   **By Jupyter Notebook cells:** It extracts and scans individual code cells from `.ipynb` files.
*   **By package.json scripts:** It extracts and scans individual commands from the `scripts` object in `package.json` files.
*   **By Markdown blocks:** It extracts and scans code snippets from triple-backtick blocks in Markdown (`.md`) files.
*   **By HTML script tags:** It extracts and scans inline code from `<script>` tags in HTML files (`.html`, `.htm`, `.xhtml`).
*   **By the first line of the file:** If a file does not have an extension, the tool checks the first line for a "shebang" (like `#!/bin/bash`). It recognizes many interpreters, including Python, Node.js, Bash (including Ash, Dash, and Zsh), Perl, Ruby, PHP, PowerShell, Lua, osascript, and iPython.
*   **Remote scripts (via URL):** It can download and scan scripts directly from the web using HTTP or HTTPS links. GitHub, GitLab, and Gist links are automatically resolved to their raw content or repository archives.

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

#### Scan Input
*   **Path to scan:** Type one or more paths (separated by spaces) or choose from the dropdown. It remembers your last 10 locations.
*   **File...:** Select one or more files to scan.
*   **Folder...:** Select a whole directory to scan.
*   **URL...:** Scan a script, Notebook, HTML, Markdown file, or archive (.zip, .tar, .tar.gz) directly from a web link.
*   **Clipboard:** Scan code currently in your clipboard.

#### Scan Options
*   **Git changes only:** Only scan files that are modified or untracked in your Git repository.
*   **Deep scan:** Scan the whole file. Normally, the scanner only checks the first and last 1 KB (1,024 bytes) of a file to save time.
*   **Scan all files:** Scan every file, even those without common script extensions.
*   **Dry Run:** Simulate the scan without actually analyzing any files.
*   **Max File Size (MB):** Skip files larger than this size.

#### AI Analysis
*   **Use AI Analysis:** Enable detailed reports for suspicious findings.
*   **Provider:** Choose between OpenAI, OpenRouter, or Ollama (local).
*   **Model:** Select the specific AI model to use (e.g., gpt-4o or llama3.2).
*   **API Base URL:** Set a custom address for your AI service.

#### Results and Filtering
*   **Filter:** Search the results list by path, threat level, notes, or code snippet.
*   **Clear:** Clear the current filter text.
*   **Min. Threat Level:** Set the sensitivity. Higher values show only the most dangerous files.
*   **Show all files:** See every scanned file, even safe ones.

#### Actions
*   **Scan Now:** Start the scanning process.
*   **Cancel:** Stop the active scan.
*   **Copy CLI Command:** Copy the command-line version of your current settings.

#### Footer Actions
*   **View:** Open a detailed view of the selected finding.
*   **Analyze with AI:** Request an AI report for the selected item(s).
*   **VirusTotal:** Look up the file's signature on VirusTotal.
*   **Open / Reveal:** Open the file or show its location in your file manager.
*   **Rescan:** Re-scan the selected file(s).
*   **Exclude:** Add the selected file(s) to your `.gptscanignore` file.
*   **Import / Export:** Save results to many formats (CSV, JSON, SARIF, HTML, Markdown) or load them from a previous scan (supports JSON, CSV, SARIF, Markdown, and HTML).
*   **Clear:** Clear all results from the list.

#### File Menu
*   **Manage Exclusions...:** View and edit your excluded file patterns.
*   **Clear AI Cache:** Delete saved AI analysis results.
*   **Clear Path History:** Wipe the list of recently scanned folders.

#### Shortcuts

**Navigation & Selection**
*   **Double-click / Enter / Space:** View detailed analysis and code.
*   **Ctrl+A / Cmd+A:** Select all results.
*   **Ctrl+F / Cmd+F:** Focus the search filter.
*   **Esc:** Cancel the active scan.

**Result Actions**
*   **Shift+Enter:** Open selected file.
*   **Ctrl+Return / Cmd+Return:** Reveal selected file in folder.
*   **F5 / R:** Rescan selected files.
*   **Ctrl+G / Cmd+G:** Analyze selected result(s) with AI.
*   **Delete:** Exclude selected results from future scans.

**Copy, Export & Import**
*   **Ctrl+C / Cmd+C:** Copy path(s) of selected result(s).
*   **Ctrl+S / Cmd+S:** Copy code snippet(s) of selected result(s).
*   **Ctrl+H / Cmd+H:** Copy SHA256 hash(es) of selected result(s).
*   **Ctrl+J / Cmd+J:** Copy selected result(s) as JSON.
*   **Ctrl+Shift+C / Cmd+Shift+C:** Copy selected result(s) as a Markdown table.
*   **Ctrl+V / Cmd+V:** Import results from clipboard.
*   **Ctrl+Shift+E / Cmd+Shift+E:** Copy the current scan settings as a CLI command.

### Using the Command Line (CLI)

Run scans from your terminal using the `--cli` flag.

**Examples:**
```bash
# Basic scan with AI analysis
python gptscan.py ./my_scripts --cli --use-gpt

# Scan multiple folders
python gptscan.py ./folder1 ./folder2 --cli

# Scan a code snippet sent from another command in the terminal
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
*   `--path [path], -p [path]`: Specify the folder or file to scan.
*   `--stdin`: Scan a code snippet from terminal input.
*   `--deep, -d`: Scan the entire file instead of just the first and last 1 KB (1,024 bytes).
*   `--dry-run`: Show which files would be scanned without analyzing them.
*   `--show-all, -a`: Show all files, including safe ones.
*   `--use-gpt, -g`: Use AI Analysis for suspicious code.
*   `--provider [name]`: Select the AI service provider (`openai`, `openrouter`, or `ollama`).
*   `--model [name]`: Set the AI model (for example: `gpt-4o`, `llama3.2`).
*   `--api-base [url]`: Set a custom URL for the AI service.
*   `--rate-limit [number]`: Limit AI requests per minute (default: 60).
*   `--clear-cache`: Clear the AI analysis cache.
*   `--output [file], -o [file]`: Save results to a file. The format depends on the extension (.json, .csv, .html, .sarif, .md).
*   `--threshold [0-100], -t [0-100]`: Set the minimum threat level to report (default: 50).
*   `--fail-threshold [0-100]`: Exit with an error if any file meets this threat level.
*   `--git-changes`: Only scan files that have changed in Git.
*   `--all-files`: Scan all files, even if they lack a script extension or a script starting line (like `#!/bin/bash`).
*   `--exclude [patterns], -e [patterns]`: Skip files that match these patterns.
*   `--file-list [file]`: Scan a list of files from a text file.
*   `--extensions [types]`: Only scan specific file types (for example: `py,js`).
*   `--import [file]`: Load results from a previous scan (JSON, CSV, SARIF, Markdown, or HTML). Use `-` for terminal input.
*   `--max-size [value]`: Set the maximum file size to scan (e.g., `10MB`, `500KB`). The default is 10MB.
*   `--json, -j`: Save or show results in JSON format (one object per line).
*   `--csv`: Save or show results in CSV format.
*   `--sarif`: Save results in SARIF format for security tools.
*   `--html`: Create an HTML report.
*   `--markdown, --md`: Create a Markdown report.
*   `--report`: Output a human-friendly triage report directly to the console, including VirusTotal links and sorted by threat level.
*   `--version, -v`: Show the tool version and exit.

## Advanced: Training

You can retrain the local scanner model to recognize new types of threats. For detailed instructions on how to prepare your data and run the trainer, see the [Training Guide](train.md).

## Troubleshooting

*   **Tkinter not found:** On Linux, run `sudo apt-get install python3-tk`.
*   **Model file missing:** Ensure `scripts.h5` is in the same folder as `gptscan.py`. This file is required for the scanner to function.
*   **Extensions list missing:** Ensure `extensions.txt` exists in the same folder. If this file is missing, the scanner will use built-in defaults (`.py`, `.js`, `.bat`, `.ps1`, `.ipynb`).
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
