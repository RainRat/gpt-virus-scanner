# GPT Virus Scanner

## Description

Can ChatGPT be used as a Virus Scanner? Yes.

This is more of a prototype than an actual product.

- Only scans scripts
- No archive scanning
- No executables scanning
- No malware removal
- No real-time scanner

## Technologies Used

- Python
- Tensorflow
- Tkinter
- ChatGPT API

## Features

- Uses built in list of file extensions to decide what to scan.
- Comes with an automatic filter, which is its own fully-functional machine-learning model, to choose which files to send to ChatGPT.
- Asks ChatGPT for an administrator's description, an end-user's description, and a threat level.
- Sort the scan results by clicking on the headers.

## Installation

- Install Python
- Add tensorflow, tkinter, openai packages to Python
- Get an OpenAI API key, put it in a file called apikey.txt
  - When you get your API key, check OpenAI's policy on data retention for yourself. I never see the contents of your files unless you send them an alternate way, but any files you send to OpenAI through your API key are associated with your account.
  
## Usage

- Run the program

```batch
python gptscan.py
```
![Scan Results](gpt-virus-scan.png)

- Show all files: List all files scanned
- Deep scan: By default, the prefilter will scan the first and last 1024 bytes of each file. Deep scan will scan every byte of the file, top to bottom.
- Use ChatGPT: Use ChatGPT to assess the interesting files.

## Contributing

- Send a pull request, and I'll add reasonable changes.
- The pre-ChatGPT filter scans the file in 1024-byte chunks, and only interesting files get sent to ChatGPT. It's also what chooses only the interesting part of the file to send.
  - The filter is a LSTM machine learning model trained on examples of clean and malware files.
  - You can contribute examples of false positive or false negatives, and I'll use them in a future update, but it's not scanning for specific malware; it might take a few dozen of a specific type of sample to get it to flip its opinion on that type. Send me a private message.
  - The 1024-byte window will be too small to see what some malware is up to. It could be increased, but it would require more training samples to fully take advantage of the increase.


## Credits

Thanks to contributors on [Stack Overflow](https://stackoverflow.com/questions/51131812/wrap-text-inside-row-in-tkinter-treeview) who I borrowed code to make the GUI. 

## License

LGPL 2.1 or later
