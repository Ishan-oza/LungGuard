import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# === CONFIG (Linux & Cross-Platform Dynamic Paths) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") if os.path.isdir(os.path.join(BASE_DIR, "models")) else BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_DIR = os.environ.get("CHX_DATASET_DIR", os.path.join(BASE_DIR, "data"))
TEST_DIR = os.path.join(DATA_DIR, "test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_INFO = {
    "resnet50": ("resnet50.pth", transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    "efficientnet_b1": ("efficientnet_b1.pth", transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    "mobilenet_v2": ("mobilenet_v2.pth", transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    "vgg19": ("vgg19.pth", transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
}

# Optimal ensemble weights (ResNet: 0.5, VGG: 0.3, EffNet: 0.1, MobileNet: 0.1)
ENSEMBLE_WEIGHTS = {
    "resnet50": 0.5,
    "vgg19": 0.3,
    "efficientnet_b1": 0.1,
    "mobilenet_v2": 0.1
}

def get_loader(transform):
    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=False)

def load_model(name, num_classes, path):
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError("Unknown model:", name)
    
    if not os.path.isfile(path):
        alt = os.path.join(BASE_DIR, f"{name}.pth")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_model(model, loader):
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def show_dashboard(y_true, y_pred, y_prob, model_name, class_names):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    # Save metrics
    with open(os.path.join(PLOTS_DIR, f"{model_name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    # Confusion Matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion.png"))
    plt.close()

    # ROC & AUC
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    per_class_auc = {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        per_class_auc[class_names[i]] = roc_auc[i]

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7,6))
    colors = ['blue','green','red']
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], color='purple', lw=2, linestyle='--', label=f"Macro-Average (AUC={roc_auc['macro']:.2f})")
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC / AUC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_roc_auc.png"))
    plt.close()

    return acc, roc_auc["macro"], per_class_auc

def weighted_ensemble_predict(model_weights):
    models_list, transforms_list, weights_list = [], [], []
    for name, w in model_weights.items():
        model = load_model(name, NUM_CLASSES, os.path.join(MODELS_DIR, MODEL_INFO[name][0]))
        transform = MODEL_INFO[name][1]
        models_list.append(model)
        transforms_list.append(transform)
        weights_list.append(w)
    loader = get_loader(transforms_list[0])
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = []
        for i, model in enumerate(models_list):
            with torch.no_grad():
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
                probs = torch.softmax(out, dim=1)
                logits.append(probs * weights_list[i])
        avg_probs = torch.stack(logits).sum(dim=0)
        preds = torch.argmax(avg_probs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(avg_probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test dataset directory '{TEST_DIR}' not found.")
        print("To benchmark all models on a test set, set CHX_DATASET_DIR environment variable.")
        exit(0)

    dataset_sample = datasets.ImageFolder(TEST_DIR, transform=MODEL_INFO["resnet50"][1])
    class_names = dataset_sample.classes

    results = []
    for model_name in MODEL_INFO.keys():
        model_path = os.path.join(MODELS_DIR, MODEL_INFO[model_name][0])
        model = load_model(model_name, NUM_CLASSES, model_path)
        loader = get_loader(MODEL_INFO[model_name][1])
        y_true, y_pred, y_prob = predict_model(model, loader)
        acc, macro_auc, per_class_auc = show_dashboard(y_true, y_pred, y_prob, model_name, class_names)
        results.append({"Model": model_name, "Accuracy": acc, "Macro_AUC": macro_auc, **per_class_auc})

    # Ensemble
    y_true, y_pred, y_prob = weighted_ensemble_predict(ENSEMBLE_WEIGHTS)
    acc, macro_auc, per_class_auc = show_dashboard(y_true, y_pred, y_prob, "ENSEMBLE", class_names)
    results.append({"Model": "ENSEMBLE", "Accuracy": acc, "Macro_AUC": macro_auc, **per_class_auc})

    # Comparison table
    df = pd.DataFrame(results)
    print("\n=== Comparison Table ===")
    print(df)
    df.to_csv(os.path.join(PLOTS_DIR, "comparison_table.csv"), index=False)

    # Bar Charts
    x = np.arange(len(df["Model"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, df["Accuracy"], width, label='Accuracy', color='skyblue')
    ax.bar(x + width/2, df["Macro_AUC"], width, label='Macro-AUC', color='salmon')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Individual Models vs Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20)
    ax.legend()
    ax.set_ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_macroauc_comparison.png"))
    plt.close()
    print("✅ Benchmarking completed and comparison table saved.")
