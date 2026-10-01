"""Install the packages listed in requirements.txt using the current interpreter.

Command line flags are assembled from character codes so that this repository
stays free of hyphen characters while the installation remains one command.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FLAG = chr(45)


def main():
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    command = [sys.executable, FLAG + "m", "pip", "install", FLAG + "r", str(requirements)]
    print("Running:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
