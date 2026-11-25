import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import itertools, os
from sklearn.metrics import accuracy_score

# === PATHS (Linux & Cross-Platform Dynamic Paths) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") if os.path.isdir(os.path.join(BASE_DIR, "models")) else BASE_DIR
DATA_DIR = os.environ.get("CHX_DATASET_DIR", os.path.join(BASE_DIR, "data"))
TEST_DIR = os.path.join(DATA_DIR, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
batch_size = 16

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(name, path):
    if name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    if not os.path.isfile(path):
        alt = os.path.join(BASE_DIR, f"{name}.pth")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Missing weights for {name} at {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test dataset directory '{TEST_DIR}' not found.")
        print("To run grid-search optimization, point CHX_DATASET_DIR to your dataset root containing 'test/'.")
        exit(0)

    test_loader = DataLoader(
        datasets.ImageFolder(TEST_DIR, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    models_dict = {
        'resnet50': load_model('resnet50', os.path.join(MODELS_DIR, 'resnet50.pth')),
        'vgg19': load_model('vgg19', os.path.join(MODELS_DIR, 'vgg19.pth')),
        'efficientnet_b1': load_model('efficientnet_b1', os.path.join(MODELS_DIR, 'efficientnet_b1.pth')),
        'mobilenet_v2': load_model('mobilenet_v2', os.path.join(MODELS_DIR, 'mobilenet_v2.pth'))
    }

    def evaluate(weights):
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                probs = torch.zeros((images.size(0), num_classes), device=device)
                for (name, model), w in zip(models_dict.items(), weights):
                    outputs = model(images)
                    probs += F.softmax(outputs, dim=1) * w
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
        return accuracy_score(y_true, y_pred)

    best_acc = 0
    best_weights = None

    print("\n🔍 Running Grid Search over Ensemble Simplex Weights...")
    for weights in itertools.product([0.1, 0.2, 0.3, 0.4, 0.5], repeat=len(models_dict)):
        if abs(sum(weights) - 1.0) > 0.01:
            continue
        acc = evaluate(weights)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
        print(f"Weights {weights} → Accuracy {acc*100:.2f}%")

    print("\n🎯 Best Accuracy:", best_acc * 100)
    print("🏆 Best Weights (resnet50, vgg19, efficientnet_b1, mobilenet_v2):", best_weights)
