# GPT Virus Scanner

Scan your scripts and files for dangerous code using AI. This tool uses a pre-trained machine learning model and optional AI analysis (OpenAI, OpenRouter, or Ollama) to find threats in scripts, Notebooks, zip files, and project files.

![GPT Virus Scanner](gpt-virus-scan.png)

## Features
*   **Scan Local & Web Files:** Scan files on your computer or directly from a web link.
*   **Platform Support:** Scan repositories and code changes from GitHub, GitLab, and Bitbucket (including PRs and `.diff`/`.patch` files).
*   **Notebook Support:** Analyzes `.ipynb` cells for dangerous commands.
*   **Project & Build Files:** Scans `package.json`, `composer.json`, `pyproject.toml`, `deno.json`, `deno.jsonc`, `Dockerfile`, `Makefile`, HTML, and Markdown.
*   **Unpack Zip & Tar:** Automatically opens `.zip`, `.tar`, and `.tar.gz` to scan the files inside.
*   **Two-step analysis:**
    1.  **Fast Local Scan:** A quick check finds suspicious patterns in milliseconds.
    2.  **AI Verification (Optional):** AI providers like OpenAI, OpenRouter, or Ollama give a detailed report on why a file is suspicious.
*   **Easy Interface:** Use the simple window (GUI) or the command line (CLI).
*   **Git Integration:** Scan only the files you have changed in your Git repository.
*   **Search & Filter:** Easily find specific results by name, threat level, or code.

## Installation

### What you need
*   **Python:** Version 3.9, 3.10, or 3.11. (Python 3.12 is not supported yet).
*   **Tkinter:** This is usually included with Python.
    *   On Linux (Ubuntu/Debian), you may need to run: `sudo apt install python3-tk`
*   (Optional) An API key for [OpenAI](https://platform.openai.com/) or [OpenRouter](https://openrouter.ai/).
*   (Optional) [Ollama](https://ollama.com/) for local AI analysis.

### How to set up
1.  Clone the repository:
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  Install the required libraries:
    ```bash
    python3 -m pip install "tensorflow<2.16" numpy openai
    ```

> **Note:** Always run the script from inside its own folder so it can find its data files (like `scripts.h5`).

## Usage

### Simple Window Mode (GUI)
Run `python3 gptscan.py` to open the scanner window.

#### What to scan
Use the **Browse** menu at the top to choose what to check:
*   **Select File(s)...:** Choose one or more scripts.
*   **Select Folder...:** Scan an entire folder.
*   **Scan URL...:** Scan a file, Notebook, or GitHub/GitLab/Bitbucket PR directly from a link (supports `package.json`, `pyproject.toml`, etc.).
*   **Scan Clipboard:** Scan code you have copied.
*   **Scan Git Diff:** Scan the changes you are currently making in your Git project.

#### Scan Options
*   **Git changes only:** Only scan files you have modified or haven't tracked yet.
*   **Deep scan:** Scan the whole file. By default, it only checks the start and end to save time.
*   **Scan all files:** Check every file, even if it doesn't look like a script.
*   **Dry Run:** See which files would be scanned without actually analyzing them.
*   **Max File Size (MB):** Skip files that are too large.

#### AI Analysis
*   **Use AI Analysis:** Get detailed reports from an AI about suspicious files.
*   **Provider:** Choose between OpenAI, OpenRouter, or Ollama (local).
*   **Model:** Select the AI model to use (like gpt-4o).
*   **API Base URL:** Use this if you have a custom AI server address.

#### Results and Filtering
*   **Filter:** Search the results for a specific name or word.
*   **Clear:** Remove the search text.
*   **Min. Threat Level:** Set how sensitive the scanner should be.
*   **Show all files:** See every file scanned, even the safe ones.

#### Actions
Use the **Results** menu at the bottom to manage your reports:
*   **Scan Now:** Start scanning.
*   **Cancel:** Stop the scan.
*   **Copy CLI Command:** Get the terminal command for your current settings.
*   **Export Results...:** Save your report as JSON, CSV, HTML, or Markdown.
*   **Import Results...:** Open a previous report.

---

### Command Line Mode (CLI)
Use the scanner in your terminal for automation.

```bash
# Scan a folder and save the result as an HTML report
python3 gptscan.py --cli ./my_scripts -o report.html

# Scan specific files using a pattern
python3 gptscan.py --cli "src/**/*.py"

# Scan only changed files in a Git project
python3 gptscan.py --cli --git-changes -t 80

# Scan a link to a GitHub Pull Request
python3 gptscan.py --cli https://github.com/user/repo/pull/123

# Scan your current Git changes
python3 gptscan.py --cli --git-diff

# Open and filter a previous report
python3 gptscan.py --cli --import results.json -o report.html
```

#### Main CLI Options
*   `--cli`: Run in the terminal instead of opening a window.
*   `target`: The folder, file, pattern, or web link to scan.
*   `--deep, -d`: Scan the whole file.
*   `--dry-run`: Show files without scanning them.
*   `--show-all, -a`: Show safe files too.
*   `--use-gpt, -g`: Use AI to explain suspicious code.
*   `--api-key [key], -k [key]`: Set your AI API key.
*   `--max-size [size]`: Skip files larger than this (e.g., "10MB").
*   `--output [file], -o [file]`: Save the results to a file.
*   `--threshold [0-100], -t [0-100]`: Only show files with this threat level or higher.
*   `--git-changes`: Only scan changed files in Git.
*   `--exclude [patterns], -e [patterns]`: Skip files that match these names.

---

## Configuration
The app saves your settings in `.gptscan_settings.json` and keeps an AI cache in `.gptscan_cache.json` to save you money on API tokens.

## License
MIT License. See the `LICENSE` file for more information.
