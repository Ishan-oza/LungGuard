#!/usr/bin/env python3
"""
=============================================================================
🫁 LungGuard - Model Weights Verification & Downloader
Downloads or verifies trained PyTorch checkpoints (.pth) in models/
=============================================================================
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1GwJ8Yt1tVA9e8vom0FzrbroRXBqL7FDt?usp=sharing"

REQUIRED_MODELS = [
    "resnet50.pth",
    "vgg19.pth",
    "efficientnet_b1.pth",
    "mobilenet_v2.pth"
]


def check_models():
    found = []
    missing = []
    search_paths = [MODELS_DIR, ROOT_DIR / "cli" / "models", ROOT_DIR / "gui" / "models"]
    for m in REQUIRED_MODELS:
        if any((p / m).is_file() for p in search_paths):
            found.append(m)
        else:
            missing.append(m)
    return found, missing


def main():
    print("=" * 64)
    print("🫁 LungGuard - Model Weights Verification & Downloader")
    print("=" * 64)

    found, missing = check_models()

    if found:
        print(f"\n✅ Found {len(found)}/{len(REQUIRED_MODELS)} models locally:")
        for m in found:
            print(f"   • {m}")

    if not missing:
        print("\n🎉 All 4 model checkpoints are present and ready for inference!")
        return

    print(f"\n⚠️  Missing {len(missing)} model weight(s):")
    for m in missing:
        print(f"   • {m}")

    print("\n" + "-" * 64)
    print("📥 DOWNLOAD INSTRUCTIONS:")
    print("Download the model weights from the official Google Drive repository:")
    print(f"👉 {GDRIVE_FOLDER_URL}")
    print(f"\nOnce downloaded, place all .pth files directly into:")
    print(f"👉 {MODELS_DIR.resolve()}")
    print("-" * 64)

    try:
        import gdown
        ans = input("\nAttempt automated download via gdown? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            print("\nDownloading folder from Google Drive...")
            gdown.download_folder(GDRIVE_FOLDER_URL, output=str(MODELS_DIR), quiet=False)
            f_after, m_after = check_models()
            if not m_after:
                print("\n🎉 All 4 models successfully installed!")
            else:
                print(f"\nNote: Please download any missing files manually from:\n{GDRIVE_FOLDER_URL}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
