# GPT Virus Scanner

## What is this?

This is a proof-of-concept security tool that checks your script files for malicious code. It works in two stages:
1.  **Local Scan:** A built-in AI model checks your files quickly.
2.  **Cloud Analysis:** If a file looks suspicious, it sends a snippet to OpenAI (ChatGPT) for a detailed report.

**Note:** This is a prototype, not a commercial antivirus product. It scans scripts (like Python, JavaScript, Batch) but not compiled executables or archives.

## Requirements

*   **Python 3.8** or newer.
*   **TensorFlow** (for the local AI model).
*   **OpenAI** (for the detailed analysis).
*   **Tkinter** (for the graphical interface).

## Installation

1.  **Get the code:**
    Clone this repository or download the files. You need `gptscan.py`, `scripts.h5`, and `task.txt` in the same folder.

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

4.  **Set up your API Key:**
    *   Get an API key from [OpenAI](https://platform.openai.com/).
    *   Create a file named `apikey.txt` in the same folder as `gptscan.py`.
    *   Paste your API key into that file (and nothing else).

    *Privacy Note:* Files are sent to OpenAI only if you enable the "Use ChatGPT" option. Check OpenAI's data policy to understand how they handle your data.

## How to Use

### Graphical Interface (GUI)

Just run the script to open the window:

```bash
python gptscan.py
```

*   **Select Directory:** Choose the folder you want to scan.
*   **Deep Scan:** Check this to scan the entire file (slower). By default, it only checks the beginning and end of files.
*   **Show all files:** Check this to see every file scanned, not just the suspicious ones.
*   **Use ChatGPT:** Check this to get a detailed report for suspicious files.

You can sort the results by clicking on the column headers.

![Scan Results](gpt-virus-scan.png)

### Command Line (CLI)

You can run scans from the terminal. This is useful for automated tasks.

**Example:**
```bash
python gptscan.py --cli --path "./my_scripts" --use-gpt
```

**Options:**
*   `--cli`: Runs in command-line mode (required).
*   `--path <folder>`: The folder to scan (required).
*   `--deep`: Scans the entire file instead of just the start and end (slower).
*   `--show-all`: Lists all files, even safe ones.
*   `--use-gpt`: Sends suspicious code to OpenAI for analysis.
*   `--extensions "py,js,bat"`: Scans these file types instead of the defaults.
*   `--rate-limit <number>`: Sets the maximum OpenAI requests per minute (default: 60).

## Contributing

We welcome improvements!

*   **False Positives/Negatives:** The local AI looks at files in small chunks (1024 bytes) and isn't perfect. If you find a file it misidentifies, please send us an example so we can retrain the model.
*   **Code:** Pull requests are welcome. Please run the tests before submitting:

    ```bash
    pip install pytest
    python -m pytest
    ```

## Credits

Thanks to the [Stack Overflow](https://stackoverflow.com/questions/51131812/wrap-text-inside-row-in-tkinter-treeview) community for the GUI code inspiration.

## License

LGPL 2.1 or later
