#!/usr/bin/env python3
"""Starts the LungGuard Web GUI by delegating to the root launcher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_gui import main

if __name__ == "__main__":
    raise SystemExit(main())
