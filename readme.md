# GPT Virus Scanner

AI-powered script analysis for local and remote files. This tool uses a pre-trained machine learning model and optional AI analysis (OpenAI, OpenRouter, or Ollama) to identify potentially malicious code in scripts, Notebooks, archives, and manifests.

![GPT Virus Scanner](gpt-virus-scan.png)

## Features
*   **Local & Remote Scanning:** Scan local files or fetch them directly from a web link.
*   **PR/MR & Patch Scanning:** Fetch and scan code changes from GitHub Pull Requests, GitLab Merge Requests, or local `.diff`/`.patch` files.
*   **Notebook Support:** Analyzes `.ipynb` cells for malicious commands.
*   **Web & Manifest Analysis:** Scans HTML, Markdown, `package.json`, `composer.json`, `deno.json`, and `deno.jsonc`.
*   **Archive Unpacking:** Automatically unpacks `.zip`, `.tar`, and `.tar.gz` to scan their contents.
*   **Dual-Stage Analysis:** 
    1.  **Fast Local Scan:** A lightweight model identifies suspicious patterns in milliseconds.
    2.  **AI Verification (Optional):** OpenRouter, OpenAI, or Ollama provides a detailed report on why a snippet is suspicious.
*   **Cross-Platform GUI & CLI:** Use the intuitive Tkinter interface or integrate it into your pipelines via terminal.
*   **Git Integration:** Option to scan only modified or untracked files in a Git repository.
*   **Advanced Filtering:** Search and filter results by path, threat level, or code content.

## Installation

### Prerequisites
*   Python 3.9, 3.10, or 3.11. (Python 3.12 is not supported yet due to model compatibility).
*   (Optional) An API key for [OpenAI](https://platform.openai.com/) or [OpenRouter](https://openrouter.ai/).
*   (Optional) [Ollama](https://ollama.com/) for local AI analysis.

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  Install dependencies:
    ```bash
    pip install "tensorflow<2.16" numpy openai pyyaml
    ```

> **Important:** Always run the script from inside its own folder so it can find its required data files (like `scripts.h5`).

## Usage

### GUI Mode
Run `python gptscan.py` to open the GUI.

#### Targets
*   **Select File...:** Choose a single script to scan.
*   **Select Folder...:** Choose a whole directory to scan.
*   **Scan URL...:** Scan a script, Notebook, HTML, Markdown file, manifest (package.json, `deno.jsonc`, etc.), PR/MR (GitHub/GitLab), or archive (.zip, .tar, .tar.gz) directly from a web link.
*   **Scan Clipboard:** Scan code currently in your clipboard.

#### Scan Options
*   **Git changes only:** Only scan files that are modified or untracked in your Git repository.
*   **Deep scan:** Scan the whole file. Normally, the scanner only checks the first and last 1 KB (1,024 bytes) of a file to save time.
*   **Scan all files:** Force the scanner to check every file, even those without common script extensions (like .txt or .log).
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
*   **Show all files:** Display every file that was scanned in the results list, including those that are safe.

#### Actions
*   **Scan Now:** Start the scanning process.
*   **Cancel:** Stop the active scan.
*   **Copy CLI Command:** Copy the command-line version of your current settings.
*   **Export Results...:** Save the report as JSON, CSV, SARIF, Markdown, or HTML.
*   **Import Results...:** Load a previous report for viewing and filtering.

---

### CLI Mode
The scanner can be used directly from the command line for automation.

```bash
# Scan a directory and save results to HTML
python gptscan.py --cli /path/to/code -o report.html

# Scan only modified files in a Git repo with high sensitivity
python gptscan.py --cli --git-changes -t 80

# Scan a remote GitHub Pull Request or GitLab Merge Request
python gptscan.py --cli https://github.com/user/repo/pull/123

# Import and filter results from a previous scan
python gptscan.py --cli --import results.json -o report.html
```

#### CLI Options
*   `--cli`: Run in command-line mode.
*   `target`: The folder, file, or web link to scan (positional argument).
*   `--path [path], -p [path]`: Alternative way to specify a folder, file, or web link to scan.
*   `--stdin`: Scan a code snippet from terminal input.
*   `--deep, -d`: Scan the entire file instead of just the first and last 1 KB (1,024 bytes).
*   `--dry-run`: Show which files would be scanned without analyzing them.
*   `--show-all, -a`: Display every file that was scanned in the results list, including those that are safe.
*   `--use-gpt, -g`: Use AI Analysis for suspicious code.
*   `--provider [name]`: Select the AI service provider (`openai`, `openrouter`, or `ollama`).
*   `--model [name]`: Set the AI model (for example: `gpt-4o`, `llama3.2`).
*   `--api-key [key], -k [key]`: Set the API key.
*   `--api-url [url]`: Set a custom base URL for the API.
*   `--max-size [MB]`: Maximum file size to scan (e.g., "10MB"). Default is 10MB.
*   `--output [file], -o [file]`: Save results to a file (JSON, CSV, SARIF, Markdown, or HTML).
*   `--report`: Output a human-friendly triage report to the console.
*   `--threshold [0-100], -t [0-100]`: Set the minimum threat level to report (default: 50).
*   `--fail-threshold [0-100]`: Exit with an error if any file meets this threat level.
*   `--git-changes`: Only scan files that have changed in Git.
*   `--all-files`: Force the scanner to check every file, even those without common script extensions (like .txt or .log) or a script starting line (like `#!/bin/bash`).
*   `--exclude [patterns], -e [patterns]`: Skip files that match these patterns.
*   `--file-list [file]`: Scan a list of files from a text file.
*   `--extensions [types]`: Only scan specific file types (for example: `py,js`).

---

## Configuration
The application stores settings in `.gptscan_settings.json` and caches AI analysis results in `.gptscan_cache.json` to save tokens.

## License
MIT License. See `LICENSE` for details.
