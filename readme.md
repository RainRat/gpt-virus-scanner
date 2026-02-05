# GPT Virus Scanner

## What is this?

This security tool scans script files for malicious code using AI. It works in two stages:
1.  **Local Scan:** A built-in AI model checks your files quickly.
2.  **Cloud Analysis:** If a file looks suspicious, it sends a snippet to an AI provider (like OpenAI) for a detailed report.

**Note:** This is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, and Batch files) but not compiled programs or archives.

## Requirements

*   **Python 3.9** or newer.
*   **TensorFlow** (version 2.15 or older).
*   **AI Provider** (Optional, for detailed analysis):
    *   **OpenAI** (requires API key)
    *   **OpenRouter** (requires API key)
    *   **Ollama** (requires local installation)
*   **Tkinter** (for the graphical interface).

## Installation

1.  **Get the code:**
    Clone this repository or download the files. Ensure `gptscan.py`, `scripts.h5`, and `task.txt` are in the same folder (these are included in the repository).

2.  **Install Python:** Download it from [python.org](https://www.python.org/).

3.  **Install required libraries:**
    Run this command in your terminal:

    ```bash
    pip install "tensorflow<2.16" openai
    ```

    *Linux users:* You may also need to install Tkinter:
    ```bash
    sudo apt-get install python3-tk
    ```

4.  **Set up your Provider (Optional):**

    If you want to use cloud analysis (OpenAI or OpenRouter), you need an API key:
    *   Create a file named `apikey.txt` in the same folder as `gptscan.py`.
    *   Paste your API key into that file.

    *   **OpenAI:** Get a key from [OpenAI](https://platform.openai.com/).
    *   **OpenRouter:** Get a key from [OpenRouter](https://openrouter.ai/).
    *   **Ollama:** No API key needed! Just ensure Ollama is running on your computer (default: `http://localhost:11434`).

    *Privacy Note:* Files are sent to the provider only if you enable the "Use AI Analysis" option. Check your provider's data policy.

## How to Use

### Graphical Interface (GUI)

Run the script to open the window:

```bash
python gptscan.py
```

*   **Select Directory:** Choose the folder you want to scan.
*   **Deep Scan:** Scan the entire file (slower). By default, the tool only checks the beginning and end of files.
*   **Show all files:** List every file scanned, including those that look safe.
*   **Use AI Analysis:** Get a detailed report for suspicious files.
*   **AI Analysis Settings:** Choose your provider (OpenAI, OpenRouter, or Ollama) and the model you want to use.
*   **Import/Export Results:** Save your scan results to CSV, JSON, HTML, or SARIF files, and load them back later for review.

You can sort the results by clicking the column headers.

![Scan Results](gpt-virus-scan.png)

### Command Line (CLI)

Run scans directly from the terminal for automated tasks.

**Example:**
```bash
python gptscan.py ./my_scripts --cli --use-gpt --json --exclude "tests/*"
```

**Options:**
*   `--cli`: Run in command-line mode (required).
*   `[target] [files...]`: The folder(s) or file(s) to scan.
*   `--path <folder>`: Specify the folder to scan.
*   `--deep`: Scan the entire file instead of just the start and end (slower).
*   `--show-all`: List all files, even safe ones.
*   `--use-gpt`: Send suspicious code to the AI provider for analysis.
*   `--json`: Output results in JSON format.
*   `--sarif`: Output results in SARIF format (standard for security tools).
*   `--html`: Output results as a standalone HTML report.
*   `--dry-run`: List files that would be scanned without running any analysis.
*   `--extensions "py,js,bat"`: Scan specific file types (comma-separated).
*   `--exclude [patterns...]`: Skip files matching these patterns (e.g., `node_modules/*`).
*   `--rate-limit <number>`: Set the maximum requests per minute (default: 60).
*   `--provider <name>`: Choose 'openai', 'openrouter', or 'ollama'.
*   `--model <name>`: Specify the model name (e.g., 'gpt-4o', 'llama3.2').
*   `--api-base <url>`: Use a custom API URL.

## Contributing

We welcome your help!

*   **Reporting issues:** If the tool misidentifies a file, please let us know.
*   **Submitting code:** Pull requests are welcome. Please run tests before submitting:

    ```bash
    pip install pytest pytest-asyncio pytest-mock
    python -m pytest
    ```

## Credits

Thanks to the [Stack Overflow](https://stackoverflow.com/questions/51131812/wrap-text-inside-row-in-tkinter-treeview) community for GUI inspiration.

## License

LGPL 2.1 or later
