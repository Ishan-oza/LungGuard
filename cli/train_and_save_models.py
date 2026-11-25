import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# === PATHS (Linux & Cross-Platform Dynamic Paths) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_DIR = os.environ.get("CHX_DATASET_DIR", os.path.join(BASE_DIR, "data"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_and_save(model_name, model, train_loader, val_loader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_losses.append(running_loss / len(train_loader))

        # === Validation ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))

        print(f"📊 [{model_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best {model_name} at {save_path}")

    # === Save Plots ===
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training Curve")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_loss.png"))
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR):
        print(f"⚠️ Training dataset folder not found at: {TRAIN_DIR}")
        print("To train models, set CHX_DATASET_DIR environment variable pointing to dataset containing 'train/'.")
        exit(0)

    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"📂 Classes detected: {class_names}")

    models_to_train = {
        "efficientnet_b1": models.efficientnet_b1(weights="IMAGENET1K_V1"),
        "mobilenet_v2": models.mobilenet_v2(weights="IMAGENET1K_V1"),
        "resnet50": models.resnet50(weights="IMAGENET1K_V1"),
        "vgg19": models.vgg19(weights="IMAGENET1K_V1")
    }

    for name, model in models_to_train.items():
        if "efficientnet" in name or "mobilenet" in name:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif "resnet" in name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "vgg" in name:
            model.classifier[6] = nn.Linear(4096, num_classes)
        model = model.to(device)
        train_and_save(name, model, train_loader, val_loader, epochs=5, lr=1e-4)
