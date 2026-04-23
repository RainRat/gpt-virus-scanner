# GPT Virus Scanner

Scan your scripts and files for dangerous code using AI. This tool uses a pre-trained machine learning model and optional AI analysis (OpenAI, OpenRouter, or Ollama) to find threats in scripts, Notebooks, zip files, and project files.

![GPT Virus Scanner](gpt-virus-scan.png)

## Features
*   **Scan Local & Web Files:** Scan files on your computer or directly from a web link.
*   **Platform Support:** Scan repositories and code changes from GitHub, GitLab, and Bitbucket (including PRs, Commits, Tags, and Snippets).
*   **Notebook Support:** Analyzes `.ipynb` cells for dangerous commands.
*   **Project & Build Files:** Scans `package.json`, `composer.json`, `pyproject.toml`, `deno.json`, `deno.jsonc`, `Dockerfile`, `Makefile`, Docker Compose (`entrypoint`), HTML, and Markdown.
*   **Unpack Zip & Tar:** Automatically opens `.zip`, `.tar`, and `.tar.gz` to scan the files inside. Supports multi-file Gists.
*   **Two-step analysis:**
    1.  **Fast Local Scan:** A quick check finds suspicious patterns in milliseconds.
    2.  **AI Verification (Optional):** AI providers like OpenAI, OpenRouter, or Ollama give a detailed report on why a file is suspicious.
*   **Easy Interface:** Use the simple window (GUI) or the command line (CLI).
*   **Git Integration:** Scan only the files you have changed in your Git repository.
*   **Search & Filter:** Easily find specific results by name, threat level, or code.

## Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you want to use the local scanner, you also need to download the model file (scripts.h5) if it's not present.*

## Usage
### GUI (Graphical Interface)
Run `python gptscan.py` to open the GUI.

Access these options from the **Browse** menu in the header:
*   **Select File(s)...:** Choose one or more scripts to scan.
*   **Select Folder...:** Choose a whole directory to scan.
*   **Scan URL...:** Scan a script, Notebook, HTML, Markdown file, Dockerfile, Makefile, manifest (package.json, `deno.jsonc`, etc.), PR/MR (GitHub/GitLab/Bitbucket), or archive (.zip, .tar, .tar.gz) directly from a web link.
*   **Scan Clipboard:** Scan code currently in your clipboard.
*   **Scan Git Diff:** Scan changes in your local Git repository.

### CLI (Command Line)
```bash
python gptscan.py path/to/your/script.py
```
You can also scan multiple files, folders, or URLs:
```bash
python gptscan.py file1.py folder/ https://github.com/user/repo/blob/main/script.py
```

### AI Verification
To use AI verification, you need an API key for OpenAI or OpenRouter, or have Ollama running locally.
1.  Open the GUI.
2.  Click the **Settings** (gear icon) in the header.
3.  Choose your provider and enter your API key or model name.

## How it works
1.  **Local Classifier:** The tool uses a Random Forest model trained on thousands of malicious and benign scripts. It looks for features like obfuscation, suspicious function calls, and dangerous strings.
2.  **AI Review:** If a file is flagged as suspicious, you can send it to an AI for a second opinion. The AI will explain *why* it thinks the code is dangerous, helping you make a final decision.

## License
This project is licensed under the GNU Lesser General Public License v2.1.
