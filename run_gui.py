#!/usr/bin/env python3
"""
LungGuard Web GUI Launcher (Cross-platform: Linux, macOS, Windows)
Usage:
    python run_gui.py
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "gui" / "backend"
APP_SCRIPT = BACKEND_DIR / "app.py"
REQ_FILE = ROOT_DIR / "requirements.txt"
REQUIRED_MODULES = ("fastapi", "uvicorn", "multipart", "torch", "torchvision", "PIL", "numpy")


def missing_modules(python_exe: str) -> list[str]:
    if Path(python_exe).resolve() == Path(sys.executable).resolve():
        return [mod for mod in REQUIRED_MODULES if importlib.util.find_spec(mod) is None]
    code = (
        "import importlib.util, json; "
        f"mods = {list(REQUIRED_MODULES)!r}; "
        "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
    )
    try:
        res = subprocess.run([python_exe, "-c", code], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout.strip())
    except OSError:
        pass
    return list(REQUIRED_MODULES)


def find_python() -> str:
    if not missing_modules(sys.executable):
        return sys.executable

    home = Path.home()
    candidates = [
        home / "anaconda3" / "bin" / "python",
        home / "miniconda3" / "bin" / "python",
        home / "anaconda3" / "python.exe",
        home / "miniconda3" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file() and not missing_modules(str(candidate)):
            return str(candidate)
    return sys.executable


def main() -> int:
    if not APP_SCRIPT.is_file():
        print(f"Error: GUI backend not found at {APP_SCRIPT}", file=sys.stderr)
        return 1

    py = find_python()
    missing = missing_modules(py)
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        try:
            ans = input(f"Install dependencies from {REQ_FILE.name} now? [Y/n]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("", "y", "yes"):
            subprocess.check_call([py, "-m", "pip", "install", "-r", str(REQ_FILE)])
        else:
            print(f'Please install dependencies first:\n  {py} -m pip install -r "{REQ_FILE}"')
            return 1

    try:
        completed = subprocess.run([py, str(APP_SCRIPT), *sys.argv[1:]], cwd=str(BACKEND_DIR))
        return completed.returncode
    except KeyboardInterrupt:
        print("\nLungGuard server stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
