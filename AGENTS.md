# Agent Guide for GPT Virus Scanner

## Project Overview
This project is a security tool that uses both local and cloud-based analysis. It combines a local TensorFlow/Keras deep learning model with an AI service (OpenAI, OpenRouter, or Ollama) to identify scripts and text files as safe or harmful. It includes a user-friendly Tkinter GUI and a robust command-line interface.

## Architecture
1. **GUI Interface:** `gptscan.py` uses `tkinter` and `ttk`. It displays results in a tabular view (Treeview).
2. **Scanner Input:** The application scans folders, files, or links. It recursively finds files in folders that match the extensions in `extensions.txt`.
3. **Stage 1 (Local Filter):**
   * The tool reads files in chunks.
   * A pre-trained Keras model (`scripts.h5`) analyzes 1024-byte windows.
   * It produces a threat level score between 0 and 100.
4. **Stage 2 (AI Analysis):**
   * If the threat level is high and the user enables AI analysis, the scanner sends the suspicious snippet to an AI provider.
   * The AI uses the prompt in `task.txt` to return an assessment containing administrator notes, end-user notes, and a threat level.

## Environment Setup
* **Python Version:** 3.9, 3.10, 3.11, or 3.12.
* **Dependencies:**
  * `tensorflow`: The machine learning library used for local scans. Make sure to install a version that is compatible with your CPU/CUDA setup.
  * `openai`: Used for AI analysis. The codebase uses the modern v1.0+ library style with `AsyncOpenAI`.
  * `tkinter`: Standard on Windows and macOS. On Linux, you must install it using your system package manager (such as `python3-tk`).
* **Required Files:**
  * `scripts.h5`: The pre-trained local model file.
  * `task.txt`: The prompt template used for AI analysis.
  * `apikey.txt`: (Optional) The API key for your AI provider.
  * `extensions.txt`: (Optional) The list of file extensions to scan.

## Code Conventions
* **Formatting:** Ensure new code changes adhere to standard PEP8 style rules.
* **Error Handling:** Keep the main GUI thread running. Use `try/except` blocks to handle exceptions in file operations and API calls.
* **Threading:** Run scanning operations in a background thread to keep the interface responsive. Send UI updates back to the main thread using a queue.

## Critical Notes for Agents
* **Security:** Do not commit `apikey.txt` or real API keys to version control.
* **Testing:** Run pytest from the repository root before submitting changes. Make sure all tests pass.
* **Coverage:** Add or update unit tests whenever you change any behavior.
* **Refactoring:** You may perform minor refactorings. For example, you can extract a block of code into a helper function to make it easier to test.
* **Model Customization and `scripts.h5`:**
  * The pre-trained `scripts.h5` binary was trained on an older version of TensorFlow in 1024-byte chunks, using ASCII 13 as filler.
  * You cannot directly modify or retrain this specific binary file.
  * However, you can train a custom model from scratch on your own dataset using `train.py`.
