import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. System Configuration
# ==========================================
DATA_DIR = "./dataset"
MODEL_SAVE_PATH = "./model/resnet18_melanoma.pth"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 2

def main():
    """
    Train and evaluate a ResNet18 model for melanoma classification.
    Implements Data Augmentation, Transfer Learning, and Dynamic LR.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Deep Learning Pipeline on: {device.type.upper()}")

    # ==========================================
    # 2. Data Augmentation & Loaders
    # ==========================================
    # Train: Apply random transformations to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Val/Test: Strictly NO random transformations, only resize and normalize
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. Model Architecture (Transfer Learning)
    # ==========================================
    # Load pre-trained ResNet18 and modify the final classification layer
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce LR by 10x if val_acc stops improving for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    # ==========================================
    # 4. Training Loop with Model Checkpointing
    # ==========================================
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("[INFO] Starting model training phase...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        # --- Validation Phase ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        # Metrics calculation
        epoch_loss = running_loss / len(train_dataset)
        train_acc = (correct_train / total_train) * 100
        val_acc = (val_correct / val_total) * 100
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        # Checkpoint mechanism: Save weights only if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  >>> Checkpoint saved! New best Val Acc: {best_val_acc:.2f}%")

    # Save the absolute best model to disk
    os.makedirs("./model", exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Best model weights saved to '{MODEL_SAVE_PATH}'")

    # ==========================================
    # 5. Final Evaluation (Test Set & ROC Curve)
    # ==========================================
    print("\n[INFO] Executing final benchmark on unseen Test Set...")
    model.eval()
    test_correct, test_total = 0, 0
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Keep probabilities for malignant class
            
    test_acc = (test_correct / test_total) * 100
    print(f"[RESULT] Deep Learning Test Accuracy: {test_acc:.2f}%")

    # ROC & AUC Calculation
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"[RESULT] ResNet18 AUC Score: {roc_auc:.4f}")

    # Render and save ROC Curve plot
    os.makedirs("./result", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ResNet18 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Deep Learning)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./result/benchmark_dl_roc.png')
    print("[SUCCESS] ROC curve saved to './result/benchmark_dl_roc.png'")

if __name__ == "__main__":
    main()