"""Forward CLI args to the project's Taichi interpreter.

Exists so allowlisted Anaconda-python invocations can launch the Taichi
environment for tests and probes:
  "D:/TOOL/Anaconda/python.exe" tools_run_taichi.py -m unittest ...
"""
from __future__ import annotations

import subprocess
import sys

TAICHI_PYTHON = r"D:\working\taichi\env\python.exe"
PROJECT_DIR = r"D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM"

sys.exit(
    subprocess.call(
        [TAICHI_PYTHON, *sys.argv[1:]],
        cwd=PROJECT_DIR,
    )
)
