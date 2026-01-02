# GPT Virus Scanner

## What is this?

This is a proof-of-concept security tool that checks your script files for malicious code. It works in two stages:
1.  **Local Scan:** A built-in AI model checks your files quickly.
2.  **Cloud Analysis:** If a file looks suspicious, it sends a snippet to an AI provider (like OpenAI) for a detailed report.

**Note:** This is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, Batch) but not compiled executables or archives.

## Requirements

*   **Python 3.8+**
*   **TensorFlow < 2.16** (Required for the local model).
*   **Tkinter** (Required for the visual interface).
*   **(Optional) AI Provider:** OpenAI, OpenRouter, or Ollama for detailed analysis.

## Installation

1.  **Download the files:**
    Keep `gptscan.py` and `scripts.h5` in the same folder.

2.  **Install Python** from [python.org](https://www.python.org/).

3.  **Install dependencies:**
    Run this in your terminal:
    ```bash
    pip install "tensorflow<2.16" openai
    ```
    *Linux users:* You may also need Tkinter: `sudo apt-get install python3-tk`

4.  **Set up AI (Optional):**
    To use AI analysis, ensure these files are in the script folder:
    *   `task.txt`: Contains instructions for the AI (included in download).
    *   `apikey.txt`: Create this file and paste your API key inside (required for OpenAI/OpenRouter).

    *Privacy Note:* Files are sent to the provider only if you enable "Use AI Analysis". Check your provider's data policy.

## How to Use

### Graphical Interface (GUI)

Run the script to start the app:

```bash
python gptscan.py
```

*   **Select Directory:** Choose the folder you want to scan.
*   **Deep Scan:** Scans the whole file (slower). Standard scan only checks the start and end.
*   **Show all files:** Lists every file scanned, not just the suspicious ones.
*   **Use AI Analysis:** Get a detailed report for suspicious files using your AI provider.
*   **Provider Settings:** Select your AI provider (OpenAI, OpenRouter, or Ollama) and model.

You can click column headers to sort the results.

![Scan Results](gpt-virus-scan.png)

### Command Line (CLI)

You can run scans from the terminal. This is useful for automation.

**Example:**
```bash
python gptscan.py ./my_scripts --cli --use-gpt --json --exclude "tests/*"
```

**Common Options:**
*   `--cli`: Run in command-line mode (required).
*   `--path <folder>`: The folder to scan.
*   `--deep`: Scan the entire file (slower).
*   `--use-gpt`: Send suspicious code to the AI for analysis.
*   `--json`: Output results as JSON (default is CSV).
*   `--sarif`: Output results in SARIF format.
*   `--dry-run`: List files that would be scanned without running the model.
*   `--exclude`: Skip files matching these patterns (e.g., `node_modules/*`).

Run `python gptscan.py --help` for a full list of options.

## Contributing

We welcome improvements!

*   **False Positives/Negatives:** The local AI scans in small chunks. If it makes a mistake, please send us the file example so we can improve the model.
*   **Code:** Pull requests are welcome. Please run the tests before submitting:

    ```bash
    pip install pytest pytest-asyncio pytest-mock
    python -m pytest
    ```

## Credits

Thanks to the [Stack Overflow](https://stackoverflow.com/questions/51131812/wrap-text-inside-row-in-tkinter-treeview) community for the GUI code inspiration.

## License

LGPL 2.1 or later
