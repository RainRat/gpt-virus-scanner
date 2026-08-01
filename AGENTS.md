# Agent Guide for GPT Virus Scanner

## Project Overview
This project is a security tool that uses both local and cloud-based analysis. It combines a local TensorFlow/Keras deep learning model with the OpenAI API to identify scripts and text files as harmful or safe. It features a Tkinter GUI.

## Architecture
1.  **GUI:** `gptscan.py` uses `tkinter` and `ttk`. It displays a Treeview of scanned files.
2.  **Input:** The user selects a folder. The app finds files in the folder and all its subfolders matching extensions in `extensions.txt`.
3.  **Stage 1 (Local Filter):**
    * Files are read in chunks.
    * A pre-trained Keras model (`scripts.h5`) analyzes 1024-byte windows.
    * It produces a threat level (`own_conf`).
4.  **Stage 2 (AI Analysis):**
    * If the local threat level is high (> 50%) and the "Use AI Analysis" checkbox is checked, the suspicious snippet is sent to the AI provider.
    * The API uses the prompt in `task.txt` to return a JSON assessment (Administrator description, End-user description, Threat Level).

## Environment Setup
* **Python Version:** Supports Python 3.9, 3.10, 3.11, or 3.12.
* **Dependencies:**
    * `tensorflow` (To ensure compatibility, install `tensorflow<2.16` on Python 3.9–3.11 or the standard `tensorflow` library without version limits on Python 3.12).
    * `openai` (Uses the modern v1.0+ API style, including asynchronous calls with `AsyncOpenAI`).
    * `tkinter` (Standard GUI library. This comes pre-installed on Windows and macOS. Linux users need to install it using their system package manager, such as `python3-tk` or `tk`).
* **Files Required for Execution:**
    * `scripts.h5`: The trained local detection model.
    * `task.txt`: The prompt templates for the AI analysis.
    * `apikey.txt`: (Optional) Local file to store your OpenAI, OpenRouter, or other provider keys.
    * `extensions.txt`: (Optional) List of file extensions to scan.

## Code Conventions
* **Formatting:** Use PEP 8 guidelines to keep your code clean and organized.
* **Error Handling:** Make sure GUI operations do not crash the main thread. Always wrap file operations, network requests, and API calls in robust `try/except` blocks.
* **Threading:** Running a scan should not lock up the user interface. Perform scan operations in a background thread and send status updates to the main thread using a queue.

## Critical Notes for Agents
* **Security:** Never save or commit `apikey.txt` or real API keys to the repository.
* **Testing:** To run tests in your local or virtual environment without TensorFlow compatibility issues on Python 3.12, run:
  ```bash
  python3 -m pytest --ignore=tests/test_train.py
  ```
  Try to fix any test failures, even if you do not think your changes caused them.
* **Unit Tests:** Always add new unit tests or update existing ones when you modify logic or add features.
* **Refactoring:** You can do small refactorings (such as moving inline code into helper functions) to make testing easier.
* **The scripts.h5 Model File:**
  The default `scripts.h5` model file was trained on an older TensorFlow version in 1024-byte windows, using ASCII 13 for padding. While you cannot retrain this exact model because the original dataset is not included in the repository, you can train a brand-new model using `train.py` on your own dataset. Rename your trained output file to `scripts.h5` and place it in the root folder to use it with `gptscan.py`.