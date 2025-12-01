# Agent Guide for GPT Virus Scanner

## Project Overview
This project is a hybrid security tool that combines a local TensorFlow/Keras deep learning model with the OpenAI API to classify scripts and text files as malicious or benign. It features a Tkinter GUI.

## Architecture
1.  **GUI:** `gptscan.py` uses `tkinter` and `ttk`. It displays a Treeview of scanned files.
2.  **Input:** The user selects a directory. The app recursively finds files matching extensions in `extensions.txt`.
3.  **Stage 1 (Local Filter):**
    * Files are read in chunks.
    * A pre-trained Keras model (`scripts.h5`) analyzes 1024-byte windows.
    * It produces a confidence score (`own_conf`).
4.  **Stage 2 (Cloud Analysis):**
    * If the local confidence is high (> 50%) and the "Use ChatGPT" checkbox is checked, the suspicious snippet is sent to the OpenAI API.
    * The API uses the prompt in `task.txt` to return a JSON assessment (Administrator description, End-user description, Threat Level).

## Environment Setup
* **Python Version:** 3.8+ recommended.
* **Dependencies:**
    * `tensorflow` (Heavy dependency, ensure compatibility with your local CUDA/CPU setup).
    * `openai` (Note: The codebase currently uses pre-1.0.0 syntax and needs upgrading).
    * `tkinter` (Usually included with Python, but may need separate install on Linux).
* **Files Required for Execution:**
    * `scripts.h5`: The trained model (binary).
    * `apikey.txt`: OpenAI API key.
    * `task.txt`: The system prompt for the LLM.
    * `extensions.txt`: List of file extensions to scan.

## Code Conventions
* **Formatting:** The current codebase is loosely formatted. New contributions should aim for PEP8 compliance.
* **Error Handling:** GUI operations should not crash the main thread. Use `try/except` blocks liberally when dealing with file I/O and API calls.
* **Threading:** The current application runs entirely on the main thread. Any long-running operations (like scanning or network requests) block the UI. **Priority should be given to moving these tasks to background threads.**

## Critical Notes for Agents
* **Security:** Do not commit `apikey.txt` or real API keys to version control.
* **Legacy Code:** You may encounter `openai.ChatCompletion`. This is deprecated. Refactor to use the `openai.OpenAI` client.
* Run `python -m pytest` from the repository root before submitting changes.
* Update or add unit tests alongside code changes when behaviour changes.
* You may do moderate refactoring if needed (ie. expose some code as a function in order to test it specifically)
* The scripts.h5 file:
    * Was trained on an older version of Tensorflow
    * Was trained in 1024-byte chunks
    * Was trained using ASCII 13 as a filler
    * You won't be able to retrain it