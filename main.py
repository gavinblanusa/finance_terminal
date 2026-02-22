"""
Launcher for Gavin Financial Terminal.

Run this file from the project root (Invest/) to start the Streamlit app.
The app code lives in app/main.py; this script invokes: streamlit run app/main.py
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/main.py", *sys.argv[1:]],
        cwd=root,
    )
