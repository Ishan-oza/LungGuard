import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

# === PATHS (Linux & Cross-Platform Dynamic Paths) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") if os.path.isdir(os.path.join(BASE_DIR, "models")) else BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
REPORT_FILE = os.path.join(BASE_DIR, "classification_report.txt")

# Test dataset directory (configured via env var or fallback)
DATA_DIR = os.environ.get("CHX_DATASET_DIR", os.path.join(BASE_DIR, "data"))
TEST_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === MODEL LOADER ===
def load_model(name, num_classes, path):
    if name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")

    if not os.path.isfile(path):
        # Fallback search
        alt_path = os.path.join(BASE_DIR, f"{name}.pth")
        if os.path.isfile(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Weights file not found at {path} or {alt_path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === ENSEMBLE WEIGHTS ===
model_weights = {
    "efficientnet_b1": 0.1,
    "mobilenet_v2": 0.1,
    "resnet50": 0.5,
    "vgg19": 0.3
}

# === PREDICTION ===
def weighted_ensemble_predict(test_loader, num_classes):
    models_dict = {
        name: load_model(name, num_classes, os.path.join(MODELS_DIR, f"{name}.pth"))
        for name in model_weights.keys()
    }

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        probs = torch.zeros((images.size(0), num_classes), device=device)

        for name, model in models_dict.items():
            outputs = model(images)
            p = torch.softmax(outputs, dim=1)
            probs += p * model_weights[name]

        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# === EVALUATION + VISUALIZATION ===
def evaluate(y_true, y_pred, y_prob, class_names):
    num_classes = len(class_names)
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n🎯 Ensemble Accuracy: {acc:.2f}%")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print("\n📊 Classification Report:\n")
    print(report_str)

    with open(REPORT_FILE, "w") as f:
        f.write(f"Ensemble Accuracy: {acc:.2f}%\n\n")
        f.write(report_str)
    print(f"📂 Report saved to: {REPORT_FILE}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📂 Confusion Matrix saved at: {cm_path}")

    # ROC-AUC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(PLOTS_DIR, "roc_auc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"📂 ROC-AUC Curve saved at: {roc_path}")

    # Save AUCs
    with open(REPORT_FILE, "a") as f:
        f.write("\n\nAUC Scores per Class:\n")
        for i in range(num_classes):
            f.write(f"{class_names[i]}: {roc_auc[i]:.3f}\n")

    # Bar Chart: Accuracy & AUC
    class_acc = [report[c]['recall'] * 100 for c in class_names]
    auc_scores = [roc_auc[i] * 100 for i in range(num_classes)]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, class_acc, width, label='Accuracy (%)')
    plt.bar(x + width/2, auc_scores, width, label='AUC (%)')
    plt.xticks(x, class_names)
    plt.ylabel("Score (%)")
    plt.title("Per-Class Accuracy vs AUC")
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(PLOTS_DIR, "accuracy_auc_bar_chart.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"📊 Bar Chart saved at: {bar_path}")

    print("✅ AUC values and plots added to report successfully!")

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test dataset directory '{TEST_DIR}' not found.")
        print("To evaluate on a full test dataset, set CHX_DATASET_DIR environment variable or place test/ folder in data/.")
        print("For single image inference, run: python predict.py")
        exit(0)

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"\n🚀 Running WEIGHTED ENSEMBLE with weights: {model_weights}")
    y_true, y_pred, y_prob = weighted_ensemble_predict(test_loader, num_classes)
    evaluate(y_true, y_pred, y_prob, class_names)
