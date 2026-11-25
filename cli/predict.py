#!/usr/bin/env python3
"""
=============================================================================
🫁 Chest X-Ray Multi-Class AI Diagnostic System (CLI)
Differential Diagnosis: COVID-19 | NORMAL | PNEUMONIA
Architecture: Optimized Soft-Voting Ensemble (ResNet50 + VGG19 + EfficientNet-B1 + MobileNetV2)
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# === CLASS DEFINITIONS ===
# Default class names matching PyTorch ImageFolder alphabetical order
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)

# Optimal ensemble weights identified via empirical grid search
DEFAULT_WEIGHTS = {
    "resnet50": 0.50,
    "vgg19": 0.30,
    "efficientnet_b1": 0.10,
    "mobilenet_v2": 0.10
}

# Image transforms
TRANSFORM_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

TRANSFORM_240 = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

MODEL_TRANSFORMS = {
    "resnet50": TRANSFORM_224,
    "vgg19": TRANSFORM_224,
    "efficientnet_b1": TRANSFORM_240,
    "mobilenet_v2": TRANSFORM_224
}


def get_default_device():
    """Detect available accelerator (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    return device, device_name


def build_model_architecture(model_name: str, num_classes: int = NUM_CLASSES):
    """Instantiate model architecture with custom classification head."""
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model


def find_model_path(model_name: str, search_dirs):
    """Search for model weights (.pth) across given directories."""
    filename = f"{model_name}.pth"
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


class CXRPredictor:
    """Manages model loading, pre-processing, and multi-model ensemble inference."""

    def __init__(self, models_dir=None, device=None, target_models=None):
        self.device, self.device_name = get_default_device() if device is None else (device, str(device))
        
        # Directory resolution
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.search_dirs = []
        env_dir = os.environ.get("LUNGGUARD_MODELS") or os.environ.get("CHESTVOTE_MODELS")
        if env_dir:
            self.search_dirs.append(env_dir)
        if models_dir:
            self.search_dirs.append(models_dir)
        self.search_dirs.extend([
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "..", "models"),
            os.path.join(base_dir, "..", "gui", "models"),
            base_dir,
            os.path.join(base_dir, "..", "pneumonia_the_final_model")
        ])

        if target_models is None:
            self.target_models = list(DEFAULT_WEIGHTS.keys())
        else:
            self.target_models = target_models

        self.models = {}
        self.active_weights = {}
        self._load_models()

    def _load_models(self):
        """Load state dicts into memory."""
        print(f"\n⚙️  Hardware Accelerator: {self.device_name} ({self.device})")
        print("🔄 Loading deep learning models...")

        total_weight = 0.0
        for name in self.target_models:
            pth_path = find_model_path(name, self.search_dirs)
            if not pth_path:
                print(f"  ❌ Model weights not found for [{name}] in search paths.")
                continue

            try:
                model = build_model_architecture(name, NUM_CLASSES)
                checkpoint = torch.load(pth_path, map_location=self.device)
                
                # Support standard state_dict or wrapped dicts
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                model.eval()
                self.models[name] = model
                w = DEFAULT_WEIGHTS.get(name, 1.0)
                self.active_weights[name] = w
                total_weight += w
                print(f"  ✅ Loaded [{name:<16}] from: {os.path.basename(pth_path)} (Weight: {w:.2f})")
            except Exception as e:
                print(f"  ⚠️ Error loading [{name}] from {pth_path}: {e}")

        if not self.models:
            raise RuntimeError(
                "No models could be loaded! Please ensure .pth files are placed in "
                f"'{self.search_dirs[0]}' or in the project root."
            )

        # Normalize active weights to sum to 1.0
        for name in self.active_weights:
            self.active_weights[name] /= total_weight
        print(f"✨ Successfully loaded {len(self.models)} active model(s).\n")

    @staticmethod
    def is_chest_xray(img):
        """Pre-scan validation: checks if input has thoracic radiograph characteristics."""
        import numpy as np
        w, h = img.size
        if min(w, h) < 100:
            return False, "Resolution too low for diagnostic analysis (minimum 100x100 required)."
        if max(w, h) / max(min(w, h), 1) > 3.2:
            return False, "Aspect ratio is non-standard for thoracic radiographs."

        img_rgb = img.convert("RGB")
        thumb = img_rgb.resize((160, 160))
        arr = np.asarray(thumb, dtype=np.float32) / 255.0

        # Saturation check (real X-rays are grayscale)
        mx = arr.max(axis=-1)
        mn = arr.min(axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
        mean_sat = float(sat.mean())
        high_sat_ratio = float((sat > 0.22).mean())
        if mean_sat > 0.14 or high_sat_ratio > 0.12:
            return False, f"Image contains vivid colors (saturation {mean_sat:.1%}). Chest X-rays are monochromatic grayscale radiographs."

        # Grayscale luminance and contrast
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        std_lum = float(gray.std())
        mean_lum = float(gray.mean())
        near_white_ratio = float((gray > 0.94).mean())
        near_black_ratio = float((gray < 0.05).mean())

        if std_lum < 0.06:
            return False, "Image lacks radiographic contrast (blank or uniform image)."
        if near_white_ratio > 0.45 or mean_lum > 0.82:
            return False, "Image appears to be a document or UI screenshot (predominantly white background)."
        if mean_lum < 0.04 and near_black_ratio > 0.85:
            return False, "Image is predominantly solid black with no thoracic structures."

        return True, "Valid chest radiograph"

    def predict_image(self, image_path: str, validate_xray: bool = True):
        """
        Run inference on a single image file.
        Returns:
            dict containing ensemble prediction, probabilities, and per-model breakdown.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image file '{image_path}': {e}")

        if validate_xray:
            is_valid, reason = self.is_chest_xray(raw_image)
            if not is_valid:
                return {
                    "image_path": image_path,
                    "is_valid_xray": False,
                    "error": reason,
                    "prediction": "INVALID",
                    "confidence": 0.0,
                    "probabilities": {c: 0.0 for c in CLASS_NAMES},
                    "per_model": {}
                }

        per_model_results = {}
        ensemble_probs = torch.zeros(NUM_CLASSES, device=self.device)

        with torch.no_grad():
            for name, model in self.models.items():
                transform = MODEL_TRANSFORMS[name]
                img_tensor = transform(raw_image).unsqueeze(0).to(self.device)
                logits = model(img_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1).squeeze(0)

                weight = self.active_weights[name]
                ensemble_probs += probs * weight

                pred_idx = torch.argmax(probs).item()
                per_model_results[name] = {
                    "pred_idx": pred_idx,
                    "pred_class": CLASS_NAMES[pred_idx],
                    "confidence": probs[pred_idx].item(),
                    "probabilities": {CLASS_NAMES[i]: probs[i].item() for i in range(NUM_CLASSES)}
                }

        ensemble_pred_idx = torch.argmax(ensemble_probs).item()
        ensemble_class = CLASS_NAMES[ensemble_pred_idx]
        ensemble_confidence = ensemble_probs[ensemble_pred_idx].item()

        return {
            "image_path": image_path,
            "prediction": ensemble_class,
            "confidence": ensemble_confidence,
            "probabilities": {CLASS_NAMES[i]: ensemble_probs[i].item() for i in range(NUM_CLASSES)},
            "per_model": per_model_results
        }


def format_progress_bar(percentage: float, width: int = 24) -> str:
    """Generate visual ASCII progress bar."""
    filled = int(round(width * percentage))
    empty = width - filled
    return "█" * filled + "░" * empty


def display_results(results: dict, show_details: bool = True):
    """Pretty-print classification results to the console."""
    if results.get("is_valid_xray") is False:
        print("\n" + "=" * 64)
        print("⚠️  PRE-SCAN INTERCEPTION: NON-CHEST-X-RAY DETECTED")
        print("=" * 64)
        print(f"  File   : {os.path.basename(results['image_path'])}")
        print(f"  Reason : {results['error']}")
        print("  Status : Model inference aborted to protect diagnostic integrity.")
        print("  Notice : Please upload a standard PA/AP chest radiograph.")
        print("           (Pass --skip-validation to bypass this filter)")
        print("=" * 64 + "\n")
        return

    pred = results["prediction"]
    conf = results["confidence"] * 100
    probs = results["probabilities"]

    # Badges
    badges = {
        "COVID19": ("🚨", "COVID-19 (SARS-CoV-2 Viral Pneumonia)", "\033[91m"),
        "PNEUMONIA": ("⚠️ ", "PNEUMONIA (Bacterial/Non-COVID Infection)", "\033[93m"),
        "NORMAL": ("✅", "NORMAL (Healthy Lung Fields)", "\033[92m")
    }
    reset_color = "\033[0m"
    icon, label, color = badges.get(pred, ("ℹ️", pred, ""))

    print("\n" + "=" * 64)
    print(f"🫁 DIAGNOSTIC INFERENCE RESULT: {os.path.basename(results['image_path'])}")
    print("=" * 64)
    print(f"  PREDICTION : {icon} {color}{label}{reset_color}")
    print(f"  CONFIDENCE : {conf:.2f}%")
    print("-" * 64)
    print(f"  {'Condition':<14} {'Probability':<12} {'Confidence Bar'}")
    print("-" * 64)

    for cls in CLASS_NAMES:
        p = probs[cls]
        p_pct = p * 100
        bar = format_progress_bar(p, width=20)
        marker = " 👈 (Top Match)" if cls == pred else ""
        print(f"  {cls:<14} {p_pct:>6.2f}%       [{bar}]{marker}")

    if show_details and results["per_model"]:
        print("\n" + "-" * 64)
        print("  MODEL BREAKDOWN & SOFT VOTES")
        print("-" * 64)
        for model_name, info in results["per_model"].items():
            m_pred = info["pred_class"]
            m_conf = info["confidence"] * 100
            print(f"  • {model_name:<16} ➔ {m_pred:<12} ({m_conf:>5.2f}% confidence)")

    print("=" * 64 + "\n")


def clean_input_path(user_input: str) -> str:
    """Strip bounding quotes and expand home directory."""
    cleaned = user_input.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    cleaned = os.path.expanduser(cleaned)
    return cleaned


def interactive_mode(predictor: CXRPredictor, show_details: bool = True):
    """Continuous interactive loop prompting user for image paths."""
    print("=" * 64)
    print("   🫁 CHEST X-RAY MULTI-CLASS AI DIAGNOSTIC SYSTEM")
    print("   Interactive Testing Mode (Enter 'q' or 'exit' to quit)")
    print("=" * 64)

    while True:
        try:
            user_input = input("👉 Enter Chest X-Ray image path: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        cleaned_path = clean_input_path(user_input)
        if cleaned_path.lower() in ("q", "quit", "exit"):
            print("Exiting session. Goodbye!")
            break

        if not cleaned_path:
            continue

        if not os.path.isfile(cleaned_path):
            print(f"❌ File not found: '{cleaned_path}'. Please check the path and try again.\n")
            continue

        try:
            result = predictor.predict_image(cleaned_path)
            display_results(result, show_details=show_details)
        except Exception as e:
            print(f"❌ Error during inference: {e}\n")

        # Ask to continue
        try:
            cont = input("Test another image? [Y/n]: ").strip().lower()
            if cont in ("n", "no", "q", "quit"):
                print("Exiting session. Goodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Multi-Class Pneumonia & COVID-19 CLI Predictor"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="Path to a chest X-ray image file (.jpg, .jpeg, .png)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "resnet50", "vgg19", "efficientnet_b1", "mobilenet_v2"],
        help="Select model to use (default: ensemble)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Custom directory path containing .pth model weights"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Hide individual model breakdown in the output"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-scan verification of chest radiograph characteristics"
    )

    args = parser.parse_args()

    # Determine which models to load
    target_models = None if args.model == "ensemble" else [args.model]

    # Initialize predictor
    try:
        predictor = CXRPredictor(
            models_dir=args.models_dir,
            target_models=target_models
        )
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        sys.exit(1)

    # Route: Single image argument vs Interactive mode
    if args.image:
        cleaned_path = clean_input_path(args.image)
        if not os.path.isfile(cleaned_path):
            print(f"❌ File not found: {cleaned_path}")
            sys.exit(1)
        res = predictor.predict_image(cleaned_path, validate_xray=not args.skip_validation)
        display_results(res, show_details=not args.no_details)
    else:
        interactive_mode(predictor, show_details=not args.no_details)


if __name__ == "__main__":
    main()
