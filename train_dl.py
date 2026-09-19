import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = "./dataset"
MODEL_SAVE_PATH = "./model/resnet18_melanoma.pth"
RESULT_DIR = "./result"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 2
SEED = 42

# ImageNet statistics, matching the pretrained weights
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    """Return accuracy (%), labels and malignant probabilities."""
    model.eval()
    labels_all, probs_all = [], []
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return 100 * correct / len(loader.dataset), labels_all, probs_all


def save_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ResNet18 (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Deep Learning)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULT_DIR, "benchmark_dl_roc.png"))
    return roc_auc


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    # Random flips and rotation for training only
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    # ImageFolder sorts class folders, so benign = 0 and malignant = 1
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Pretrained ResNet18 with a new 2-class head
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cut the LR by 10x when val accuracy has not improved for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        val_acc, _, _ = evaluate(model, val_loader, device)
        train_acc = 100 * correct / len(train_dataset)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | LR {lr:.6f} | Loss {running_loss / len(train_dataset):.4f} "
              f"| Train {train_acc:.2f}% | Val {val_acc:.2f}%")
        scheduler.step(val_acc)

        # Keep the weights with the best val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  new best val accuracy: {best_val_acc:.2f}%")

    model.load_state_dict(best_weights)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSaved best weights to {MODEL_SAVE_PATH}")

    test_acc, labels, probs = evaluate(model, test_loader, device)
    roc_auc = save_roc_curve(labels, probs)
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
