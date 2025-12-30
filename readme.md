# GPT Virus Scanner

## What is this?

This is a proof-of-concept security tool that checks your script files for malicious code. It works in two stages:
1.  **Local Scan:** A built-in AI model checks your files quickly.
2.  **Cloud Analysis:** If a file looks suspicious, it sends a snippet to an AI provider (like OpenAI) for a detailed report.

**Note:** This is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, Batch) but not compiled executables or archives.

## Requirements

*   **Python 3.8** or newer.
*   **TensorFlow** (for the local AI model).
*   **LLM Provider** (Optional, for detailed analysis):
    *   **OpenAI** (requires API key)
    *   **OpenRouter** (requires API key)
    *   **Ollama** (requires local installation)
*   **Tkinter** (for the graphical interface).

## Installation

1.  **Get the code:**
    Clone this repository or download the files. You need `gptscan.py` and `scripts.h5` in the same folder.
    *   To enable AI analysis, you also need a `task.txt` file containing the system prompt for the AI.

2.  **Install Python** from [python.org](https://www.python.org/).

3.  **Install the required libraries:**
    Run this command in your terminal. We specify an older TensorFlow version to match our AI model.

    ```bash
    pip install "tensorflow<2.16" openai
    ```

    *Linux users:* You might also need to install Tkinter:
    ```bash
    sudo apt-get install python3-tk
    ```

4.  **Set up your Provider (Optional):**

    If you want to use cloud analysis (OpenAI or OpenRouter), you need an API key:
    *   Create a file named `apikey.txt` in the same folder as `gptscan.py`.
    *   Paste your API key into that file (and nothing else).

    *   **OpenAI:** Get a key from [OpenAI](https://platform.openai.com/).
    *   **OpenRouter:** Get a key from [OpenRouter](https://openrouter.ai/).
    *   **Ollama:** No API key needed! Just ensure Ollama is running locally (default: `http://localhost:11434`).

    *Privacy Note:* Files are sent to the provider only if you enable the "Use AI Analysis" option. Check your provider's data policy.

## How to Use

### Graphical Interface (GUI)

Just run the script to open the window:

```bash
python gptscan.py
```

*   **Select Directory:** Choose the folder you want to scan.
*   **Deep Scan:** Check this to scan the entire file (slower). By default, it only checks the beginning and end of files.
*   **Show all files:** Check this to see every file scanned, not just the suspicious ones.
*   **Use AI Analysis:** Check this to get a detailed report for suspicious files.
*   **Provider Settings:** Choose between OpenAI, OpenRouter, or Ollama, and specify the model.

You can sort the results by clicking on the column headers.

![Scan Results](gpt-virus-scan.png)

### Command Line (CLI)

You can run scans from the terminal. This is useful for automated tasks.

**Example:**
```bash
python gptscan.py ./my_scripts --cli --use-gpt --json
```

**Options:**
*   `--cli`: Runs in command-line mode (required).
*   `[target]`: The file or folder to scan (positional argument).
*   `--path <folder>`: Alternative way to specify the folder to scan.
*   `--deep`: Scans the entire file instead of just the start and end (slower).
*   `--show-all`: Lists all files, even safe ones.
*   `--use-gpt`: Sends suspicious code to the LLM for analysis.
*   `--json`: Outputs results in JSON format (default is CSV).
*   `--sarif`: Outputs results in SARIF format (standard for security tools).
*   `--dry-run`: Lists files that would be scanned without running the AI model.
*   `--extensions "py,js,bat"`: Scans these file types instead of the defaults.
*   `--rate-limit <number>`: Sets the maximum requests per minute (default: 60).
*   `--provider <name>`: Choose 'openai', 'openrouter', or 'ollama' (default: openai).
*   `--model <name>`: Specify the model name (e.g., 'gpt-4o', 'llama3.2').
*   `--api-base <url>`: Set a custom API URL.

## Contributing

We welcome improvements!

*   **False Positives/Negatives:** The local AI looks at files in small chunks (1024 bytes) and isn't perfect. If you find a file it misidentifies, please send us an example so we can retrain the model.
*   **Code:** Pull requests are welcome. Please run the tests before submitting:

    ```bash
    pip install pytest pytest-asyncio pytest-mock
    python -m pytest
    ```

## Credits

Thanks to the [Stack Overflow](https://stackoverflow.com/questions/51131812/wrap-text-inside-row-in-tkinter-treeview) community for the GUI code inspiration.

## License

LGPL 2.1 or later
