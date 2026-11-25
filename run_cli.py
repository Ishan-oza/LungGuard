#!/usr/bin/env python3
"""
LungGuard CLI Launcher (Cross-platform: Linux, macOS, Windows)
Usage:
    python run_cli.py
    python run_cli.py -i cli/samples/sample_chest_xray.png
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = ROOT_DIR / "cli" / "predict.py"
REQUIRED_MODULES = ("torch", "torchvision", "PIL", "numpy")


def has_dependencies(python_exe: str) -> bool:
    if Path(python_exe).resolve() == Path(sys.executable).resolve():
        return all(importlib.util.find_spec(mod) is not None for mod in REQUIRED_MODULES)
    check = (
        "import importlib.util, sys; "
        f"sys.exit(0 if all(importlib.util.find_spec(m) for m in {REQUIRED_MODULES!r}) else 1)"
    )
    try:
        res = subprocess.run([python_exe, "-c", check], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except OSError:
        return False


def find_python() -> str:
    if has_dependencies(sys.executable):
        return sys.executable

    home = Path.home()
    candidates = [
        home / "anaconda3" / "bin" / "python",
        home / "miniconda3" / "bin" / "python",
        home / "anaconda3" / "python.exe",
        home / "miniconda3" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file() and has_dependencies(str(candidate)):
            return str(candidate)
    return sys.executable


def main() -> int:
    if not CLI_SCRIPT.is_file():
        print(f"Error: CLI entry point not found at {CLI_SCRIPT}", file=sys.stderr)
        return 1

    py = find_python()
    if not has_dependencies(py):
        print("Missing required packages (torch, torchvision, Pillow, numpy).")
        print(f'Install them with:\n  {py} -m pip install -r "{ROOT_DIR / "requirements.txt"}"')
        return 1

    try:
        completed = subprocess.run([py, str(CLI_SCRIPT), *sys.argv[1:]])
        return completed.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
