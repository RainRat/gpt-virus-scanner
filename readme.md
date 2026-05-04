# GPT Virus Scanner

Scan your files for dangerous code with AI. This tool uses a quick scan model to find threats in scripts, Notebooks, archives, and project files.

![GPT Virus Scanner](gpt-virus-scan.png)

## Features

### Core Capabilities
*   **Two-step analysis:**
    1.  **Fast Local Scan:** A quick check finds suspicious patterns in milliseconds.
    2.  **AI Analysis (Optional):** Get a detailed report from OpenAI, OpenRouter, or Ollama explaining why a file is suspicious.
*   **Flexible Interface:** Use the friendly window interface or the command line for automation.
*   **Git Integration:** Scan only the files you have changed in your project.
*   **Search & Filter:** Easily find specific results by name, threat level, or code.
*   **Unified Diffs:** Scan `.diff` and `.patch` files to review code changes.

### Supported Platforms
*   **Scan Local & Web Files:** Scan files on your computer or directly from a web link.
*   **Remote Repositories:** Scan code from GitHub, GitLab, and Bitbucket (including PRs, Commits, and Tags).
*   **Web Snippets:** Scan from Pastebin and Hugging Face.

### File Format Support
*   **Notebook Support:** Scan cells in `.ipynb` files for dangerous commands.
*   **Project & Build Files:** Scan `package.json`, `composer.json`, `pyproject.toml`, `deno.json`, `deno.jsonc`, `Dockerfile`, `Makefile`, and Docker Compose.
*   **Archives:** Open `.zip`, `.tar`, and `.tar.gz` files automatically to scan the contents.
*   **CI/CD Workflows:** Scan GitHub Actions, GitLab CI, and other YAML workflows for suspicious commands.
*   **Web Files:** Scan HTML and Markdown files for embedded scripts.

## What you need
*   **Python:** You need **Python 3.9, 3.10, or 3.11**. Newer versions (like 3.12) are not supported yet.
*   **Data files:** You need the `scripts.h5` model file and `task.txt` AI instructions in the project folder. Both are included in this project.

## How to install
1.  **Clone the project:**
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  **Install the required packages:**
    ```bash
    python3 -m pip install "tensorflow<2.16" openai numpy
    ```
    *Note: `pyyaml` is also needed if you plan to train your own models. If you are on Linux, you may also need `python3-tk` for the window interface.*

## How to use
### Using the Window (GUI)
Run `python3 gptscan.py` to open the scanner window.

Access these options from the **Browse** menu:
*   **Select File(s)...:** Choose one or more scripts to scan.
*   **Select Folder...:** Choose a whole directory to scan.
*   **Scan URL...:** Scan scripts or archives directly from a web link.
*   **Scan File List...:** Read a list of files to scan from a text file.
*   **Scan Clipboard:** Scan code currently in your clipboard.
*   **Scan Git Diff:** Scan changes in your local project.
*   **Scan Git Revision...:** Scan files modified in a specific Git revision or commit.

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
The scanner provides several ways to analyze and manage your results:
*   **Search & Filter:** Use the **Filter** bar (or press `Ctrl+F`) to search through results by file path, threat level, or code.
*   **View Details:** Double-click any result or press `Space` to see a detailed analysis and the full source code.
*   **Right-Click Menu:** Right-click a result to access quick actions, such as **Check on VirusTotal** or **View Online**.
*   **Export Reports:** Save your results in various formats (CSV, Markdown, HTML, JSON, or SARIF) by selecting **File > Export Results...**.

## Customizing the Scanner
You can tailor the scanner to your needs:
*   **Exclusions:** Ignore specific files or folders by using **File > Manage Exclusions...** or by adding patterns to a `.gptscanignore` file.
*   **Extensions:** Control which file types are scanned by using **File > Manage Extensions...** or by editing the `extensions.txt` file.

## Advanced Usage
### Training the Model
You can train the local "quick scan" model on your own data. This requires `pyyaml` to be installed.
See [Training the Local Scanner](train.md) for more information.

## How it works
1.  **Local Filter:** The tool uses a quick scan model trained on thousands of safe and dangerous scripts. It looks for patterns like hidden code and suspicious commands.
2.  **AI Analysis:** If a file looks suspicious, you can ask an AI for a second opinion. The AI will explain *why* it thinks the code is dangerous, helping you decide what to do.

## License
This project is licensed under the GNU Lesser General Public License v2.1.
