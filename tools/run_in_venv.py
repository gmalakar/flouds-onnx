# =============================================================================
# File: run_in_venv.py
# Date Created: 2026-11-26
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Run a Python command with the repository's .venv interpreter.

Used by pre-commit hooks that need packages installed in .venv (for example
``python -E tools/run_in_venv.py -m pyright``). The hook may start under any
``python`` on PATH; ``-E`` keeps a PYTHONHOME inherited from a uv launcher of a
different version from crashing it. The command runs as a child process (not
``os.execv``, which on Windows ends this process before the child finishes)
without PYTHONHOME, and its exit code is returned.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def venv_python() -> Path:
    root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def main(argv: list[str]) -> int:
    interpreter = venv_python()
    if not interpreter.exists():
        print(f"{interpreter} not found: create the virtualenv first.", file=sys.stderr)
        return 1
    env = {key: value for key, value in os.environ.items() if key != "PYTHONHOME"}
    return subprocess.run([str(interpreter), *argv], env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
