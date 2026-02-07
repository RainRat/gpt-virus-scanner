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
4.  **Stage 2 (AI Analysis):**
    * If the local confidence is high (> 50%) and the "Use AI Analysis" checkbox is checked, the suspicious snippet is sent to the AI provider.
    * The API uses the prompt in `task.txt` to return a JSON assessment (Administrator description, End-user description, Threat Level).

## Environment Setup
* **Python Version:** 3.8+ recommended.
* **Dependencies:**
    * `tensorflow` (Heavy dependency, ensure compatibility with your local CUDA/CPU setup).
    * `openai` (v1.0+ syntax used, including `AsyncOpenAI`).
    * `tkinter` (Usually included with Python, but may need separate install on Linux).
* **Files Required for Execution:**
    * `scripts.h5`: The trained model (binary).
    * `task.txt`: The system prompt for the LLM.
    * `apikey.txt`: (Optional) OpenAI/Provider API key.
    * `extensions.txt`: (Optional) List of file extensions to scan.

## Code Conventions
* **Formatting:** The current codebase is loosely formatted. New contributions should aim for PEP8 compliance.
* **Error Handling:** GUI operations should not crash the main thread. Use `try/except` blocks liberally when dealing with file I/O and API calls.
* **Threading:** Scanning operations run on a background thread to keep the UI responsive. UI updates are marshaled back to the main thread via a queue.

## Critical Notes for Agents
* **Security:** Do not commit `apikey.txt` or real API keys to version control.
* Always run `pytest` from the repository root before submitting changes, unless it is a documentation-only change. Try to fix any test failures, even if you don't think you caused them.
* Update or add unit tests alongside code changes when behaviour changes.
* You may do moderate refactoring if needed (ie. expose some code as a function in order to test it specifically)
* The scripts.h5 file:
    * Was trained on an older version of Tensorflow
    * Was trained in 1024-byte chunks
    * Was trained using ASCII 13 as a filler
    * You won't be able to retrain it