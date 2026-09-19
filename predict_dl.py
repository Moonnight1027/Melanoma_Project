import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Same preprocessing as training (FOV mask + DullRazor + resize)
from prepare_data import advanced_preprocess

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

TEST_DIR = "./test"
RESULT_DIR = "./result"
MODEL_PATH = "./model/resnet18_melanoma.pth"
CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match eval_transforms in train_dl.py
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def predict_image(image_path, model):
    """Classify one raw image and save it with the prediction in the title."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return

    rgb = cv2.cvtColor(advanced_preprocess(img), cv2.COLOR_BGR2RGB)
    x = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    confidence, pred = torch.max(probs, 0)
    label = CATEGORIES[pred.item()]
    print(f"{os.path.basename(image_path)}: {label} ({confidence.item() * 100:.2f}%)")

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"ResNet18: {label}\nConfidence: {confidence.item() * 100:.2f}%")
    plt.axis("off")
    plt.tight_layout()

    name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(os.path.join(RESULT_DIR, f"dl_pred_{name}.png"))
    plt.close()


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_dl.py first.")
        return
    model = load_model()

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(IMAGE_EXTS)] if os.path.exists(TEST_DIR) else []
    if not files:
        print(f"No images found in {TEST_DIR}")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    for file_name in files:
        predict_image(os.path.join(TEST_DIR, file_name), model)
    print(f"\nFigures saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
