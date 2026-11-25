"""Model loading + ensemble inference + Grad-CAM for LungGuard."""
import base64, io, os, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]          # ImageFolder alphabetical order
LABELS = {"COVID19": "COVID-19", "NORMAL": "Normal", "PNEUMONIA": "Pneumonia"}
WEIGHTS = {"resnet50": 0.5, "vgg19": 0.3, "efficientnet_b1": 0.1, "mobilenet_v2": 0.1}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HERE = Path(__file__).resolve().parent


def _tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def build(name, n=3):
    if name == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "vgg19":
        m = models.vgg19(weights=None); m.classifier[6] = nn.Linear(4096, n)
    elif name == "efficientnet_b1":
        m = models.efficientnet_b1(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise ValueError(name)
    return m


def model_dirs():
    env = os.environ.get("LUNGGUARD_MODELS") or os.environ.get("CHESTVOTE_MODELS")
    dirs = [Path(env)] if env else []
    # Look in backend/models, parent models, cli models
    dirs += [HERE / "models", HERE.parent / "models", HERE.parent.parent / "cli" / "models",
             HERE.parent.parent / "cli", HERE.parent.parent / "models"]
    return dirs


class Engine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models, self.paths, self.missing = {}, {}, []
        self.lock = threading.Lock()

    def load(self):
        for name in WEIGHTS:
            path = next((d / f"{name}.pth" for d in model_dirs() if (d / f"{name}.pth").is_file()), None)
            if not path:
                self.missing.append(name); continue
            try:
                ck = torch.load(path, map_location=self.device)
                if isinstance(ck, dict) and "state_dict" in ck: ck = ck["state_dict"]
                m = build(name); m.load_state_dict(ck); m.to(self.device).eval()
                self.models[name], self.paths[name] = m, str(path)
            except Exception as e:  # keep serving with whatever loaded
                self.missing.append(name); print(f"[warn] could not load {name}: {e}")
        return self

    @property
    def ready(self):
        return bool(self.models)

    def info(self):
        return {"ready": self.ready, "device": str(self.device), "loaded": list(self.models),
                "missing": self.missing, "search_paths": [str(d) for d in model_dirs()[:5]]}

    # ---------- Pre-scan validation: verify input is a genuine chest X-ray
    @staticmethod
    def is_chest_xray(img):
        """
        Validates whether the uploaded image has the optical and radiographic
        characteristics of a posteroanterior/anteroposterior (PA/AP) chest X-ray.
        Rejects color images, UI screenshots, text documents, or blank files.
        """
        w, h = img.size
        if min(w, h) < 100:
            return False, "Resolution is too low for radiographic evaluation (minimum 100x100 required)."
        if max(w, h) / max(min(w, h), 1) > 3.2:
            return False, "Image aspect ratio is non-standard for thoracic radiographs."

        img_rgb = img.convert("RGB")
        thumb = img_rgb.resize((160, 160))
        arr = np.asarray(thumb, dtype=np.float32) / 255.0  # (160, 160, 3)

        # 1. Color saturation check (genuine radiographs are monochromatic grayscale)
        mx = arr.max(axis=-1)
        mn = arr.min(axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
        mean_sat = float(sat.mean())
        high_sat_ratio = float((sat > 0.22).mean())

        if mean_sat > 0.14 or high_sat_ratio > 0.12:
            return False, f"Image contains vivid colors (color saturation {mean_sat:.1%}). Chest X-rays are monochromatic grayscale radiographs."

        # 2. Grayscale luminance and contrast check
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        std_lum = float(gray.std())
        mean_lum = float(gray.mean())
        near_white_ratio = float((gray > 0.94).mean())
        near_black_ratio = float((gray < 0.05).mean())

        if std_lum < 0.06:
            return False, "Image lacks radiographic contrast (blank or uniform image)."

        # Reject screenshots, white documents, receipts, web pages
        if near_white_ratio > 0.45 or mean_lum > 0.82:
            return False, "Image appears to be a document or screenshot (predominantly white background)."

        if mean_lum < 0.04 and near_black_ratio > 0.85:
            return False, "Image is predominantly solid black with no visible thoracic anatomical structures."

        return True, "Valid chest radiograph"

    @staticmethod
    def check_image(img):
        warns = []
        w, h = img.size
        if min(w, h) < 200:
            warns.append("Resolution is below 200px. Fine pulmonary features may be obscured.")
        return warns

    @staticmethod
    def _colormap(v):  # v in 0..1 -> RGB, matches the palette
        stops = np.array([[0, 104, 59, 43], [.35, 176, 132, 1], [.7, 212, 158, 141], [1, 250, 246, 242]], dtype=np.float32)
        return np.stack([np.interp(v, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], -1)

    def _gradcam(self, img, class_idx):
        """Grad-CAM on ResNet50 layer4. Returns transparent RGBA PNG (data URI) at image aspect."""
        m = self.models["resnet50"]
        x = _tf(224)(img).unsqueeze(0).to(self.device)
        acts, grads = {}, {}
        def fwd(_, __, out):
            acts["a"] = out; out.register_hook(lambda g: grads.__setitem__("g", g))
        h = m.layer4.register_forward_hook(fwd)
        try:
            with torch.enable_grad():
                m.zero_grad(); out = m(x); out[0, class_idx].backward()
        finally:
            h.remove()
            m.zero_grad(set_to_none=True)
        w = grads["g"].mean((2, 3), keepdim=True)
        cam = F.relu((w * acts["a"]).sum(1, keepdim=True)).detach()
        W, H = img.size; s = 640 / max(W, H, 640)
        size = (max(int(H * s), 1), max(int(W * s), 1))
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.cpu().numpy()
        rgb = self._colormap(cam)
        alpha = (np.clip(cam, 0, 1) ** 1.4 * 190).astype(np.uint8)
        out_img = Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]))
        buf = io.BytesIO(); out_img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---------- main entry
    def predict(self, img, which="ensemble", explain=True):
        if not self.ready: raise RuntimeError("No model weights loaded.")
        img = img.convert("RGB")
        t0 = time.perf_counter()

        # Pre-scan verification: check if input image is a chest X-ray
        is_valid, validation_msg = self.is_chest_xray(img)
        if not is_valid:
            return {
                "is_valid_xray": False,
                "error": validation_msg,
                "model": which,
                "prediction": "INVALID",
                "label": "Non-Thoracic Image",
                "confidence": 0.0,
                "probabilities": {k: 0.0 for k in CLASS_NAMES},
                "per_model": {},
                "heatmap": None,
                "warnings": [validation_msg],
                "inference_ms": round((time.perf_counter() - t0) * 1000),
                "image": {"width": img.size[0], "height": img.size[1]}
            }

        names = list(self.models) if which == "ensemble" else [which]
        if which != "ensemble" and which not in self.models: raise KeyError(which)
        total = sum(WEIGHTS[n] for n in names)
        ens = torch.zeros(len(CLASS_NAMES), device=self.device)
        per = {}
        with self.lock:
            with torch.no_grad():
                for n in names:
                    size = 240 if (n == "efficientnet_b1" and which != "ensemble") else 224
                    p = torch.softmax(self.models[n](_tf(size)(img).unsqueeze(0).to(self.device)), 1)[0]
                    wt = WEIGHTS[n] / total; ens += p * wt
                    i = int(p.argmax())
                    per[n] = {"prediction": CLASS_NAMES[i], "label": LABELS[CLASS_NAMES[i]], "confidence": float(p[i]),
                              "weight": round(wt, 3), "probabilities": {c: float(p[j]) for j, c in enumerate(CLASS_NAMES)}}
            top = int(ens.argmax())
            heat = None
            if explain and "resnet50" in self.models:
                try: heat = self._gradcam(img, top)
                except Exception as e: print("[warn] gradcam failed:", e)
        c = CLASS_NAMES[top]
        return {
            "is_valid_xray": True,
            "error": None,
            "model": which,
            "prediction": c,
            "label": LABELS[c],
            "confidence": float(ens[top]),
            "probabilities": {k: float(ens[j]) for j, k in enumerate(CLASS_NAMES)},
            "per_model": per,
            "heatmap": heat,
            "warnings": self.check_image(img),
            "inference_ms": round((time.perf_counter() - t0) * 1000),
            "image": {"width": img.size[0], "height": img.size[1]}
        }
