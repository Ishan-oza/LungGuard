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
MODEL_NAME = "resnet50"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_resnet50(num_classes, path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if not os.path.isfile(path):
        alt = os.path.join(BASE_DIR, "resnet50.pth")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Missing resnet50.pth at {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test dataset directory '{TEST_DIR}' not found.")
        print("To evaluate on a test set, set CHX_DATASET_DIR to the dataset root folder.")
        print("For single image inference, run: python predict.py -m resnet50")
        exit(0)

    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_names = dataset.classes

    model = load_resnet50(NUM_CLASSES, os.path.join(MODELS_DIR, "resnet50.pth"))

    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            out = model(images)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {MODEL_NAME} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)
