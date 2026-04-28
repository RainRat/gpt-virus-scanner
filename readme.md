# GPT Virus Scanner

Scan your files for dangerous code with AI. This tool uses a machine learning model to find threats in scripts, Notebooks, archives, and project files.

![GPT Virus Scanner](gpt-virus-scan.png)

## Features
*   **Scan Local & Web Files:** Scan files on your computer or from a web link.
*   **Platform Support:** Scan code from GitHub, GitLab, Bitbucket, Gists, Pastebin, and Hugging Face (including PRs, Commits, Tags, and Snippets).
*   **Notebook Support:** Scan cells in `.ipynb` files for dangerous commands.
*   **Project & Build Files:** Scan `package.json`, `composer.json`, `pyproject.toml`, `deno.json`, `deno.jsonc`, `Dockerfile`, `Makefile`, Docker Compose, HTML, and Markdown.
*   **Unpack Archives:** Open `.zip`, `.tar`, and `.tar.gz` files automatically to scan the contents.
*   **Two-step analysis:**
    1.  **Fast Local Scan:** A quick check finds suspicious patterns in milliseconds.
    2.  **AI Analysis (Optional):** Get a detailed report from OpenAI, OpenRouter, or Ollama explaining why a file is suspicious.
*   **Easy Interface:** Use the graphical window or the command line.
*   **Git Integration:** Scan only the files you have changed in your Git repository.
*   **Search & Filter:** Easily find specific results by name, threat level, or code.

## What you need
*   **Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet.
*   **Included files:** The repository already includes the `scripts.h5` model file and the `task.txt` AI instructions. You do not need to download these separately.

## How to install
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  **Install the required packages:**
    ```bash
    python3 -m pip install "tensorflow<2.16" openai numpy pyyaml
    ```
    *Note: If you are on Linux, you may also need to install `python3-tk` for the window interface.*

## How to use
### Using the Window (GUI)
Run `python3 gptscan.py` to open the scanner window.

Access these options from the **Browse** menu:
*   **Select File(s)...:** Choose one or more scripts to scan.
*   **Select Folder...:** Choose a whole directory to scan.
*   **Scan URL...:** Scan a script, Notebook, HTML, Markdown file, Dockerfile, Makefile, manifest (package.json, `deno.jsonc`, etc.), PR/MR (GitHub/GitLab/Bitbucket), Pastebin paste, Hugging Face blob, or archive (.zip, .tar, .tar.gz) directly from a web link.
*   **Scan File List...:** Read a list of files to scan from a text file.
*   **Scan Clipboard:** Scan code currently in your clipboard.
*   **Scan Git Diff:** Scan changes in your local Git repository.

### Using the Terminal (CLI)
To run the scanner in your terminal, use the `--cli` flag:
```bash
python3 gptscan.py path/to/your/script.py --cli
```
You can also scan multiple files, folders, or web links:
```bash
python3 gptscan.py file1.py folder/ https://github.com/user/repo --cli
```

### Setting up AI Analysis
To use AI analysis, you need an API key for OpenAI or OpenRouter, or have Ollama running locally.
1.  Open the GUI.
2.  In the **AI Analysis** panel, check the **Use AI Analysis** box.
3.  Choose your provider and enter your API key or model name.

## Reviewing Results
Once a scan is complete, you can interact with the results in the table:
*   **Filter results:** Press `Ctrl+F` to quickly search for specific files, threat levels, or suspicious code.
*   **See details:** Double-click a row or press `Space` to see a detailed analysis of why a file was flagged.
*   **Right-click menu:** Right-click any result to access more tools:
    *   **View Online:** Open the file on GitHub, GitLab, or Bitbucket.
    *   **VirusTotal:** Check the file's hash against the VirusTotal database.
    *   **Exclude Selected:** Ignore specific files or patterns in future scans.
    *   **Copy:** Quickly copy the file path, hash, or code snippet.
*   **Export reports:** Use the **Results** button in the bottom corner (or `Ctrl+E`) to save your results as a JSON, CSV, HTML, or Markdown report.

## Customizing the Scanner
*   **Ignore files:** Create a `.gptscanignore` file in the project folder to exclude specific files or folders using glob patterns (like `node_modules/*` or `*.log`).
*   **Change file types:** Edit `extensions.txt` to add or remove the file extensions the scanner looks for (for example, `.php` or `.rb`).

## How it works
1.  **Local Filter:** The tool uses a deep learning model trained on thousands of safe and dangerous scripts. It looks for patterns like hidden code and suspicious commands.
2.  **AI Analysis:** If a file looks suspicious, you can ask an AI for a second opinion. The AI will explain *why* it thinks the code is dangerous, helping you decide what to do.

## License
This project is licensed under the GNU Lesser General Public License v2.1.
