# 🫁 LungGuard: AI-Assisted Chest Radiograph Differential Diagnosis System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA Accelerated](https://img.shields.io/badge/Compute-CUDA%20%7C%20CPU-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.97%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/Macro%20ROC--AUC-0.994-gold.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **LungGuard** is a deep learning clinical decision-support system that evaluates posteroanterior (PA) chest radiographs (CXRs) for multi-class differential diagnosis across **COVID-19**, **Pneumonia (Viral & Bacterial)**, and **Normal Healthy Lungs**. Powered by an empirically weighted soft-voting ensemble of four heterogeneous neural architectures with Grad-CAM explainability heatmaps, pre-scan non-X-ray validation, and dual deployment interfaces (Production Web GUI + Standalone CLI).

---

## 📑 Table of Contents

- [Overview & Clinical Motivation](#-overview--clinical-motivation)
- [Why LungGuard is Different & Better](#-why-lungguard-is-different--better)
- [Model Architecture & Consensus Voting](#-model-architecture--consensus-voting)
- [Model Weights Download](#-model-weights-download)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
  - [Linux (Ubuntu / Debian / Arch / Fedora)](#linux-ubuntu--debian--arch--fedora)
  - [macOS (Apple Silicon & Intel)](#macos-apple-silicon--intel)
  - [Windows (PowerShell & Command Prompt)](#windows-powershell--command-prompt)
- [Quick Start: Running LungGuard](#-quick-start-running-lungguard)
  - [1. Web Graphical User Interface (GUI)](#1-web-graphical-user-interface-gui)
  - [2. Command-Line Interface (CLI)](#2-command-line-interface-cli)
- [Evaluation Benchmarks & Results](#-evaluation-benchmarks--results)
- [Developer REST API Reference](#-developer-rest-api-reference)
- [Project Directory Structure](#-project-directory-structure)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🏥 Overview & Clinical Motivation

In clinical pulmonology and emergency medicine, timely and accurate triage of pulmonary infections is critical. While viral pneumonia, bacterial pneumonia, and COVID-19 share overlapping clinical presentations (cough, dyspnea, fever), their treatment trajectories differ fundamentally:

- **COVID-19 (SARS-CoV-2)** requires rapid isolation, antiviral therapies, and oxygen monitoring.
- **Bacterial Pneumonia** mandates targeted antibacterial regimens.
- **Normal Radiographs** prevent inappropriate antimicrobial administration, reducing antimicrobial resistance.

Standard chest radiograph interpretation is subject to inter-observer variability, especially in high-volume emergency triage environments. **LungGuard** addresses this challenge by providing immediate, explainable, and reproducible multi-class triage support entirely within local memory.

```
                  ┌──────────────────────────────────────────────┐
                  │          Uploaded Chest Radiograph           │
                  └──────────────────────┬───────────────────────┘
                                         │
                         [ Pre-Scan Optical Filter ]
                         (Validates Grayscale & Thoracic Contrast)
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
             ┌──────▼──────┐                           ┌──────▼──────┐
             │  ResNet50   │ (Weight: 0.50)            │    VGG19    │ (Weight: 0.30)
             └──────┬──────┘                           └──────┬──────┘
                    │                                         │
             ┌──────▼──────┐                           ┌──────▼──────┐
             │EfficientNet │ (Weight: 0.10)            │ MobileNetV2 │ (Weight: 0.10)
             └──────┬──────┘                           └──────┬──────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                        [ Weighted Soft-Voting Consensus ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         Diagnostic Probability                   Grad-CAM Attention
         [COVID-19 | Normal | Pneumonia]           Pulmonary Heatmap
```

---

## 🌟 Why LungGuard is Different & Better

Most open-source pneumonia detection repositories on GitHub implement simple binary classification ("Normal" vs "Pneumonia") on toy datasets with a single generic network. **LungGuard** was engineered specifically to solve real-world clinical and engineering limitations:

| Clinical & Engineering Dimension | Standard Open-Source Detectors | LungGuard Multi-Class Ensemble |
| :--- | :--- | :--- |
| **Diagnostic Scope** | Binary only (*Normal* vs *Pneumonia*) | **3-Class Differential Diagnosis** (*COVID-19*, *Normal*, *Pneumonia*) |
| **Model Architecture** | Single standalone CNN (prone to overfitting) | **4-Architecture Ensemble** (*ResNet50*, *VGG19*, *EfficientNet-B1*, *MobileNetV2*) |
| **Consensus Formulation** | Naive average or unweighted majority | **Empirically Optimized Soft-Voting** (0.50 / 0.30 / 0.10 / 0.10) |
| **Clinical Explainability** | Black box (no visual reasoning) | **Grad-CAM Attention Heatmaps** overlaid directly on radiograph |
| **Input Safeguard Gate** | None (evaluates cats, cars, selfies as X-rays) | **Pre-Scan Heuristic Gate** intercepts non-chest images before inference |
| **Deployment Interfaces** | Raw Jupyter notebook or monolithic script | **Dual Mode**: Editorial Web GUI + Standalone Scriptable CLI |
| **Patient Privacy** | Transmits scans to third-party cloud APIs | **100% In-Memory Processing** (Zero disk writes, total privacy) |
| **Cross-Platform** | Linux-only or complex Docker dependencies | **Native Support for Linux, macOS, and Windows** |

---

## 🧠 Model Architecture & Consensus Voting

To prevent model-specific inductive bias, LungGuard synthesizes predictions from four structurally distinct deep learning paradigms:

1. **ResNet50 (Residual Learning)**: Employs identity shortcut connections to eliminate vanishing gradients across 50 deep layers, serving as the high-confidence clinical anchor (Weight: `0.50`).
2. **VGG19 (Deep Homogeneous Representation)**: Utilizes sequential $3 \times 3$ convolutional filters to capture fine-grained interstitial patterns and alveolar consolidations (Weight: `0.30`).
3. **EfficientNet-B1 (Compound Scaling)**: Jointly scales depth, width, and resolution using neural architecture search (NAS) and mobile inverted bottlenecks (Weight: `0.10`).
4. **MobileNetV2 (Inverted Residuals & Linear Bottlenecks)**: Lightweight, high-throughput model providing rapid regularization against false positives (Weight: `0.10`).

### Mathematical Formulation of Weighted Soft-Voting

For an input radiograph $X$ and classes $c \in \{\text{COVID-19}, \text{Normal}, \text{Pneumonia}\}$:

$$P_{\text{ensemble}}(c \mid X) = \sum_{m=1}^{4} w_m \cdot P_m(c \mid X)$$

$$\hat{y} = \arg\max_{c} P_{\text{ensemble}}(c \mid X)$$

where weights $w = [0.50, 0.30, 0.10, 0.10]$ satisfy $\sum w_m = 1.0$, optimized via multi-dimensional grid search across held-out validation partitions.

---

## 📥 Model Weights Download

Because trained PyTorch model checkpoints exceed GitHub's single-file upload thresholds, the weights are distributed through official Google Drive storage:

### 🔗 **[Click Here to Download Model Checkpoints (.pth)](https://drive.google.com/drive/folders/1GwJ8Yt1tVA9e8vom0FzrbroRXBqL7FDt?usp=sharing)**

| Model Filename | Architecture | Checkpoint Size | Target Directory |
| :--- | :--- | :--- | :--- |
| `resnet50.pth` | ResNet-50 | ~94.4 MB | `models/` |
| `vgg19.pth` | VGG-19 | ~558.3 MB | `models/` |
| `efficientnet_b1.pth` | EfficientNet-B1 | ~26.5 MB | `models/` |
| `mobilenet_v2.pth` | MobileNetV2 | ~9.2 MB | `models/` |

> 💡 **Where to place them:** Simply drop the four downloaded `.pth` files into the `models/` directory in the root of the cloned repository. Both CLI and GUI applications will discover them automatically.

---

## 💻 System Requirements

- **Python**: Version 3.8 to 3.12 (Python 3.10+ recommended)
- **RAM**: Minimum 4 GB (8 GB+ recommended for multi-model loading)
- **Disk Space**: ~1 GB free space for model weights and code
- **Hardware Acceleration**:
  - **CUDA GPU**: NVIDIA GPU with 4GB+ VRAM (Auto-detected; inference takes ~40–90 ms)
  - **Apple Silicon**: Metal Performance Shaders (MPS) auto-supported on macOS
  - **CPU**: Fully supported across all x86_64 and ARM64 processors (inference takes ~250–600 ms)

---

## 🚀 Installation Guide

### Linux (Ubuntu / Debian / Arch / Fedora)

```bash
# 1. Clone the repository
git clone https://github.com/Ishan-oza/LungGuard
cd lungguard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify model weights
python3 download_models.py
```

### macOS (Apple Silicon & Intel)

```bash
# 1. Clone the repository
git clone https://github.com/Ishan-oza/LungGuard
cd lungguard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (PyTorch with MPS acceleration)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify model weights
python3 download_models.py
```

### Windows (PowerShell & Command Prompt)

```powershell
# 1. Clone the repository
git clone https://github.com/Ishan-oza/LungGuard
cd lungguard

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify model weights
python download_models.py
```

---

## 🎯 Quick Start: Running LungGuard

LungGuard offers two native execution interfaces:

### 1. Web Graphical User Interface (GUI)

The GUI provides an editorial clinical workstation styled in warm earth tones, featuring real-time laser scan animation, Grad-CAM attention heatmap inspection, condition probability breakdown, and evaluation benchmark charts.

**To launch on Linux, macOS, or Windows:**
```bash
python run_gui.py
```

Or run the backend module directly:
```bash
cd gui/backend
python app.py
```
Open **`http://localhost:8000`** in any web browser.

#### GUI Key Features:
- 🖼️ **Drag-and-Drop Radiograph Ingestion**: Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`.
- ⚡ **Animated Diagnostic Beam**: Visual top-to-bottom laser scan feedback while models execute.
- 🔬 **Grad-CAM Activation Map Overlay**: High-resolution pulmonary attention heatmaps to localize opacities.
- 🛡️ **Pre-Scan Verification**: Rejects selfies, document screenshots, or non-medical images before neural network execution.
- 📊 **Interactive Model Inspector**: Inspect each model's individual vote alongside the ensemble consensus.

---

### 2. Command-Line Interface (CLI)

The CLI tool is self-contained and ideal for batch scripting, server automation, or headless environments without a browser.

**Interactive Mode (Linux, macOS, or Windows):**
```bash
# From repo root
python run_cli.py
# Or:
python cli/predict.py
```
You will be prompted to enter the path to a radiograph image, and results will be rendered in formatted ASCII tables with colored clinical status indicators.

**Direct Argument Inference:**
```bash
python cli/predict.py -i cli/samples/sample_chest_xray.png
```

**Select a Specific Single Model:**
```bash
python cli/predict.py -i cli/samples/sample_chest_xray.png -m resnet50
```

**CLI Output Example:**
```text
================================================================
🫁 DIAGNOSTIC INFERENCE RESULT: sample_chest_xray.png
================================================================
  PREDICTION : ⚠️  PNEUMONIA (Bacterial/Non-COVID Infection)
  CONFIDENCE : 44.75%
----------------------------------------------------------------
  Condition      Probability  Confidence Bar
----------------------------------------------------------------
  COVID19         38.25%       [████████░░░░░░░░░░░░]
  NORMAL          17.00%       [███░░░░░░░░░░░░░░░░░]
  PNEUMONIA       44.75%       [█████████░░░░░░░░░░░] 👈 (Top Match)

----------------------------------------------------------------
  MODEL BREAKDOWN & SOFT VOTES
----------------------------------------------------------------
  • resnet50         ➔ PNEUMONIA    (64.93% confidence)
  • vgg19            ➔ COVID19      (62.05% confidence)
  • efficientnet_b1  ➔ COVID19      (77.61% confidence)
  • mobilenet_v2     ➔ COVID19      (99.60% confidence)
================================================================
```

---

## 📊 Evaluation Benchmarks & Results

LungGuard was evaluated on an independent test cohort of **1,929 unseen patient radiographs** with zero overlap with training data.

### Overall Performance Summary
- **Ensemble Accuracy**: `94.97%` (1,832 / 1,929 correct calls)
- **Macro-Averaged F1 Score**: `0.94`
- **Macro-Averaged Area Under the ROC Curve (AUC)**: `0.994`

### Per-Condition Diagnostic Metrics

| Clinical Condition | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Test Cohort Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | **0.88** | **0.97** | **0.92** | **0.995** | 332 scans |
| **Normal (Healthy)** | **0.97** | **0.94** | **0.95** | **0.996** | 529 scans |
| **Pneumonia (Non-COVID)** | **0.97** | **0.95** | **0.96** | **0.990** | 1,068 scans |
| **Macro Average** | **0.94** | **0.95** | **0.94** | **0.994** | **1,929 scans** |

> 💡 **Clinical Safety Highlight**: The ensemble maintains a **97.3% recall on COVID-19**, ensuring that contagious viral cases are flagged with minimal risk of false negatives.

### Confusion Matrix (1,929 Test Scans)

| Actual Condition \ Predicted | COVID-19 (Predicted) | Normal (Predicted) | Pneumonia (Predicted) |
| :--- | :---: | :---: | :---: |
| **COVID-19 (Actual)** | **323** *(97.3%)* | 6 *(1.8%)* | 3 *(0.9%)* |
| **Normal (Actual)** | 0 *(0.0%)* | **497** *(94.0%)* | 32 *(6.0%)* |
| **Pneumonia (Actual)** | 45 *(4.2%)* | 11 *(1.0%)* | **1,012** *(94.8%)* |

---

## 📡 Developer REST API Reference

LungGuard includes a high-performance **FastAPI** service suitable for integrating into hospital PACS, research workstations, or telemedicine microservices.

### `POST /api/predict`
Executes multi-class inference and generates Grad-CAM attention heatmaps.

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -F "file=@cli/samples/sample_chest_xray.png" \
     -F "model=ensemble" \
     -F "explain=true"
```

**JSON Response:**
```json
{
  "is_valid_xray": true,
  "error": null,
  "model": "ensemble",
  "prediction": "PNEUMONIA",
  "label": "Pneumonia",
  "confidence": 0.4475,
  "probabilities": {
    "COVID19": 0.3825,
    "NORMAL": 0.1700,
    "PNEUMONIA": 0.4475
  },
  "per_model": {
    "resnet50": { "prediction": "PNEUMONIA", "label": "Pneumonia", "confidence": 0.6493, "weight": 0.5 },
    "vgg19": { "prediction": "COVID19", "label": "COVID-19", "confidence": 0.6205, "weight": 0.3 },
    "efficientnet_b1": { "prediction": "COVID19", "label": "COVID-19", "confidence": 0.7761, "weight": 0.1 },
    "mobilenet_v2": { "prediction": "COVID19", "label": "COVID-19", "confidence": 0.9960, "weight": 0.1 }
  },
  "heatmap": "data:image/png;base64,iVBORw0KGgoAAA...",
  "warnings": [],
  "inference_ms": 78,
  "image": { "width": 1024, "height": 1024 }
}
```

### `GET /api/health`
Returns active hardware acceleration and loaded neural network weights.

### `GET /api/metrics`
Returns test dataset evaluation statistics, confusion matrices, and per-class metrics.

---

## 📂 Project Directory Structure

```text
lungguard/
├── .gitignore                   # Excludes heavy .pth weights, cache, and logs
├── README.md                    # Comprehensive documentation and clinical guide
├── requirements.txt             # Consolidated project dependencies
├── download_models.py           # Verification and helper script for model weights
├── run_cli.py                   # Cross-platform CLI launcher (Linux / macOS / Windows)
├── run_gui.py                   # Cross-platform GUI launcher (Linux / macOS / Windows)
├── models/                      # Folder for trained model checkpoints
│   ├── .gitkeep
│   └── README.md                # Download instructions and Google Drive link
├── cli/                         # Standalone Command-Line Tool
│   ├── predict.py               # Main CLI predictor (interactive + argument modes)
│   ├── samples/                 # Sample test radiograph
│   │   └── sample_chest_xray.png
│   ├── classification_report.txt # Raw empirical test metrics
│   └── plots/                   # Loss curves and ROC curves
└── gui/                         # Web Graphical User Interface & API
    ├── run.py                   # Cross-platform GUI starter
    ├── backend/                 # FastAPI REST Engine
    │   ├── app.py               # API routes and webserver
    │   ├── inference.py         # PyTorch ensemble engine & Grad-CAM generator
    │   ├── requirements.txt     # Backend specific requirements
    │   └── data/
    │       └── metrics.json     # 1,929 patient cohort evaluation data
    └── frontend/                # Medical AI User Interface
        ├── index.html           # Overview and clinical rationale
        ├── analyze.html         # Diagnostic radiograph workbench
        ├── models.html          # Backbone architecture technical details
        ├── results.html         # Confusion matrix and ROC-AUC benchmarks
        ├── method.html          # Pipeline formulation & API documentation
        └── assets/
            ├── style.css        # Responsive editorial theme with Soria typography
            ├── app.js           # Client-side validation, scan animation, & charts
            ├── logo.svg         # Clean LungGuard brand vector
            └── fonts/
                └── soria.ttf    # Dani Bydani's official Soria serif font
```

---

## ⚖️ Medical Disclaimer

> **IMPORTANT CLINICAL NOTICE**: LungGuard is an experimental machine learning research prototype developed for educational, benchmarking, and investigational purposes only. It is **not** a certified medical device, has not been cleared or approved by the U.S. FDA, CE, or any other health regulatory authority, and is **not intended for primary medical diagnosis, prognosis, or clinical treatment**. Diagnostic decisions must always be made by a licensed healthcare professional and board-certified radiologist in conjunction with patient history, clinical symptoms, laboratory assays, and standard-of-care imaging protocols.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
