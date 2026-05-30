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

### Supported Sources
*   **Scan Local & Web Files:** Scan files on your computer or directly from a web link.
*   **Remote Repositories:** Scan code from GitHub (including Gists), GitLab, and Bitbucket (including Snippets, PRs, Commits, and Tags).
*   **Web Snippets:** Scan from Pastebin and Hugging Face.

### File Format Support
*   **Notebook Support:** Scan cells in `.ipynb` files for dangerous commands.
*   **Project & Build Files:** Scan `package.json`, `composer.json`, `pyproject.toml`, `deno.json`, `deno.jsonc`, `Dockerfile`, `Makefile`, and Docker Compose.
*   **Archives:** Open `.zip`, `.tar`, and `.tar.gz` files automatically to scan the contents.
*   **CI/CD Workflows:** Scan GitHub Actions, GitLab CI, and other YAML workflows for suspicious commands.
*   **Web Files:** Scan HTML, SVG, and Markdown files for embedded scripts.
*   **Unified Diffs:** Scan `.diff` and `.patch` files to review code changes.

## Installation

### Prerequisites
*   **Python:** Install **Python 3.9, 3.10, or 3.11**. Newer versions like 3.12 are not yet supported.
*   **Data files:** The repository already includes the `scripts.h5` model and `task.txt` instruction files. Keep these in the project folder.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RainRat/gpt-virus-scanner.git
    cd gpt-virus-scanner
    ```
2.  **Install mandatory packages:**
    ```bash
    python3 -m pip install "tensorflow<2.16" openai numpy
    ```
3.  **Install optional packages (if needed):**
    *   **python3-tk:** Install this if you use Linux and want the window (GUI) interface.
    *   **PyYAML:** Install this if you want to train your own models.

## How to use
### Using the Window (GUI)
Run `python3 gptscan.py` to open the scanner window.

Access these options from the **Browse** menu:
*   **Scan File(s)...:** Choose one or more scripts to scan.
*   **Scan Folder...:** Choose a whole directory to scan.
*   **Scan Recently Modified...:** Scan files that have been changed within a certain timeframe (e.g., last 24 hours).
*   **Scan URL...:** Scan scripts or archives directly from a web link.
*   **Scan File List...:** Read a list of files to scan from a text file.
*   **Scan Clipboard:** Scan code currently in your clipboard.
*   **Scan Shell Profiles:** Scan your shell configuration files (.bashrc, .zshrc, etc.) and PowerShell profiles for dangerous aliases or functions (Ctrl+Shift+B).
*   **Scan Git Diff:** Scan changes in your local project.
*   **Scan Git Hooks:** Scan local and global Git hooks for dangerous scripts (Ctrl+Shift+G).
*   **Scan Git Configuration:** Scan potentially dangerous Git configuration settings (aliases, editors, etc.).
*   **Scan Git Revision...:** Scan files modified in a specific Git revision or commit.
*   **Scan System Audit:** Perform a comprehensive scan of your system including shell profiles, history, system PATH, SSH configurations, running processes, environment variables, scheduled tasks, startup items, system services, Git configuration, Git hooks, and installed Python packages (Ctrl+Shift+I).
*   **Scan Shell History:** Automatically find and scan your terminal history (Bash, Zsh, PowerShell, etc.) for dangerous one-liners.
*   **Scan System PATH:** Scan all directories in your system PATH for suspicious executables or scripts.
*   **Scan SSH Configuration:** Scan your SSH configuration and authorized_keys files for suspicious settings (Ctrl+Shift+Z).
*   **Scan Running Processes:** Scan command lines of all running processes to find potentially dangerous execution strings (Ctrl+Shift+K).
*   **Scan Environment Variables:** Scan all non-empty environment variables for suspicious scripts or commands (Ctrl+Shift+N).
*   **Scan Scheduled Tasks:** Scan all scheduled tasks (Windows) and Cron jobs (Linux/macOS) to identify dangerous persistence (Ctrl+Shift+T).
*   **Scan Startup Items:** Scan all system startup items and LaunchAgents to find malicious persistence (Ctrl+Shift+A).
*   **Scan System Services:** Scan all system services (systemd files on Linux, Service PathName on Windows) to identify dangerous persistence (Ctrl+Shift+S).
*   **Scan Python Packages:** Scan all directories containing installed Python packages (site-packages) for potentially malicious modules (Ctrl+Shift+Y).
*   **Scan Node.js Packages:** Scan all directories containing global Node.js packages for potentially malicious modules.
*   **Scan Editor Extensions:** Scan all directories containing editor extensions (VS Code, Sublime Text, Vim) for potentially malicious scripts (Ctrl+Shift+X).

### Keyboard Shortcuts
The scanner includes shortcuts for faster navigation:

| Shortcut | Action |
| :--- | :--- |
| **General** | |
| `Enter` | Start Scan |
| `Esc` | Cancel Scan |
| `Ctrl+F` | Focus Filter Bar |
| `Ctrl+O` | Import Results |
| `Ctrl+E` | Export Results |
| `Ctrl+V` | Import Results from Clipboard |
| `Ctrl+Shift+E` | Copy as Command Line |
| **Scan Actions** | |
| `Ctrl+Shift+O` | Scan File(s) |
| `Ctrl+Shift+U` | Scan URL |
| `Ctrl+Shift+V` | Scan Clipboard |
| `Ctrl+Shift+D` | Scan Git Diff |
| `Ctrl+Shift+G` | Scan Git Hooks |
| `Ctrl+Shift+B` | Scan Shell Profiles |
| `Ctrl+Shift+I` | Scan System Audit |
| `Ctrl+Shift+H` | Scan Shell History |
| `Ctrl+Shift+P` | Scan System PATH |
| `Ctrl+Shift+K` | Scan Running Processes |
| `Ctrl+Shift+N` | Scan Environment Variables |
| `Ctrl+Shift+T` | Scan Scheduled Tasks |
| `Ctrl+Shift+A` | Scan Startup Items |
| `Ctrl+Shift+S` | Scan System Services |
| `Ctrl+Shift+Y` | Scan Python Packages |
| `Ctrl+Shift+X` | Scan Editor Extensions |
| `Ctrl+Shift+Z` | Scan SSH Configuration |
| **Results List** | |
| `Space` / `Enter` | View Details |
| `F5` / `r` | Rescan |
| `Delete` | Exclude |
| `Ctrl+A` | Select All |
| `Ctrl+C` | Copy File Path |
| `Ctrl+Shift+C` | Copy as Markdown Table |
| `Ctrl+H` | Copy SHA-256 Hash |
| `Ctrl+S` | Copy Code Snippet |
| `Ctrl+J` | Copy Results as JSON |
| `Ctrl+G` | Analyze Selected with AI |
| `Shift+Enter` | Open File |
| `Ctrl+Enter` | Reveal in Folder |
| `Ctrl+T` | Check on VirusTotal |
| `Ctrl+L` | View Online |
| **Details Window** | |
| `Esc` | Close Window |
| `Left` / `Right` | Previous / Next Result |
| `F5` / `r` | Rescan |
| `Delete` | Exclude |
| `Ctrl+U` | Toggle Full Source |
| `Ctrl+S` | Copy Code Snippet |
| `Ctrl+Shift+C` | Copy AI Analysis |
| `Ctrl+J` | Copy JSON Data |
| `Ctrl+L` | View Online |
| `Shift+Enter` | Open File |
| `Ctrl+Enter` | Reveal in Folder |

*Note: macOS users should use `Command` instead of `Ctrl` for most shortcuts.*

### Using the Terminal (CLI)
To run the scanner in your terminal, use the `--cli` flag.

#### Basic Usage
Scan a single file or folder:
```bash
python3 gptscan.py path/to/your/script.py --cli
```

Scan multiple files, folders, or web links:
```bash
python3 gptscan.py file1.py folder/ https://github.com/user/repo --cli
```

Scan files modified in the last 24 hours:
```bash
python3 gptscan.py --modified 24h --cli
```

#### System Scans
Perform a comprehensive system audit:
```bash
python3 gptscan.py --audit --cli
```

Scan all directories containing installed Python packages:
```bash
python3 gptscan.py --python-packages --cli
```

Scan all directories containing global Node.js packages:
```bash
python3 gptscan.py --nodejs-packages --cli
```

Scan SSH configuration and authorized_keys files:
```bash
python3 gptscan.py --ssh-config --cli
```

Scan all directories containing editor extensions:
```bash
python3 gptscan.py --editor-extensions --cli
```

Scan all common shell profile and RC files:
```bash
python3 gptscan.py --shell-profiles --cli
```

Scan your terminal history (Bash, Zsh, PowerShell, etc.):
```bash
python3 gptscan.py --shell-history --cli
```

Scan all directories in your system PATH:
```bash
python3 gptscan.py --system-path --cli
```

Scan command lines of all running processes:
```bash
python3 gptscan.py --running-processes --cli
```

Scan all scheduled tasks and Cron jobs:
```bash
python3 gptscan.py --scheduled-tasks --cli
```

Scan all system startup items and LaunchAgents:
```bash
python3 gptscan.py --startup-items --cli
```

Scan all system services:
```bash
python3 gptscan.py --system-services --cli
```

Scan all environment variables:
```bash
python3 gptscan.py --env-vars --cli
```

#### Git Integration
Scan changes in your local project as a diff:
```bash
python3 gptscan.py --git-diff --cli
```

Scan local and global Git hooks for dangerous scripts:
```bash
python3 gptscan.py --git-hooks --cli
```

Scan potentially dangerous Git configuration settings:
```bash
python3 gptscan.py --git-config --cli
```

#### Advanced Scans
Scan code sent from another command in the terminal:
```bash
echo "import os; os.system('rm -rf /')" | python3 gptscan.py --stdin --cli
```

Save scan results to a file (CSV, JSON, HTML, etc.):
```bash
python3 gptscan.py ./my_project --output results.html --cli
```

### Setting up AI Analysis
To use AI analysis, you need an API key for OpenAI or OpenRouter, or have Ollama running locally.

#### API Keys
You can provide your API key in three ways:
*   **In the GUI:** Enter it in the **AI Analysis** panel. It will be saved locally to `apikey.txt`.
*   **Environment Variables:** Set the `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable in your terminal.
*   **Local File:** Create a file named `apikey.txt` in the project folder and paste your key there.

*Note: Do not share `apikey.txt` or commit it to a public repository.*

#### Custom API Base (Advanced)
If you use a local proxy or a custom endpoint (like a specific Ollama setup or an OpenAI-compatible server), you can set a custom **API Base** URL in the GUI or with the `--api-base` terminal flag.

## Reviewing Results
The scanner provides several ways to analyze and manage your results:
*   **Filtering Results:**
    *   **Filter Bar:** Use the **Filter** bar at the top (or press `Ctrl+F`) to quickly find results by file path, analysis text, or code snippets.
    *   **Min. Threat Level:** Use the **Min. Threat Level** setting in the **Filter** bar to hide files with low threat scores.
*   **View Details:** Double-click any result or press `Space` to see a detailed analysis and the full source code.
*   **Right-Click Menu:** Right-click a result for quick actions:
    *   **Rescan:** Scan the file again (useful after making changes).
    *   **Exclude:** Add the file or folder to your ignore list.
    *   **Check on VirusTotal:** Search for the file's hash on VirusTotal.
    *   **View Online:** Open the source file in your web browser (for Git projects and remote URLs).
*   **Export & Import:**
    *   **Export Results:** Save your scan to a file (CSV, Markdown, HTML, JSON, or SARIF) via **File > Export Results...**.
    *   **Import Results:** Load previous scan results from any of the supported formats via **File > Import Results...** or by pasting them from your clipboard (`Ctrl+V`).

## Customizing the Scanner
You can tailor the scanner to your needs:
*   **Exclusions:** Ignore specific files or folders by using **File > Manage Exclusions...** or by adding patterns to a `.gptscanignore` file. In the terminal, use the `-e` or `--exclude` flag.
*   **Extensions:** Control which file types are scanned by using **File > Manage Extensions...** or by editing the `extensions.txt` file. In the terminal, use the `--extensions` flag.
*   **File Size:** The scanner skips files larger than 10MB during folder scans. You can adjust this limit in the **Scan Options** panel or by using the `--max-size` flag. Files you select individually are always scanned, regardless of their size.
*   **Deep Scan:** Scan the entire file instead of just the beginning and end. This is more thorough but slower. Use the **Deep Scan** checkbox or the `-d` or `--deep` flag.
*   **Scan All Files:** By default, the scanner only checks script-like files (like `.py` or `.js`). Use the **Scan All Files** checkbox or the `--all-files` flag to check every file.
*   **Dry Run:** Preview which files would be scanned without actually checking them. Use the **Dry Run** checkbox or the `--dry-run` flag.

## Advanced Usage
### Training the Model
You can train the local "quick scan" model on your own data. This requires `pyyaml` to be installed.
See [Training the Local Scanner](train.md) for more information.

## How it works
1.  **Local Filter:** The tool uses a quick scan model trained on thousands of safe and dangerous scripts. It looks for patterns like hidden code and suspicious commands.
2.  **AI Analysis:** If a file looks suspicious, you can ask an AI for a second opinion. The AI will explain *why* it thinks the code is dangerous, helping you decide what to do.

## License
This project is licensed under the GNU Lesser General Public License v2.1.
