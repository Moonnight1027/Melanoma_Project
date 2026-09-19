import os

import cv2
import joblib
import matplotlib.pyplot as plt

# Same preprocessing and features as training
from ext import IMG_SIZE, extract_features, segment_lesion
from prepare_data import advanced_preprocess

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

TEST_DIR = "./test"
RESULT_DIR = "./result"
MODEL_PATH = "./model/melanoma_model.pkl"
SCALER_PATH = "./model/feature_scaler.pkl"
CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def predict_image(image_path, model, scaler):
    """Classify one raw image and save the image with its lesion mask."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return

    img = cv2.resize(advanced_preprocess(img), (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(img)
    features = scaler.transform(extract_features(img).reshape(1, -1))

    probs = model.predict_proba(features)[0]
    pred = int(probs.argmax())
    label = CATEGORIES[pred]
    print(f"{os.path.basename(image_path)}: {label} ({probs[pred] * 100:.2f}%)")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Preprocessed (FOV + DullRazor)\nDiag: {label}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Region of Interest Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(os.path.join(RESULT_DIR, f"xgb_pred_{name}.png"))
    plt.close()


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train.py first.")
        return
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(IMAGE_EXTS)] if os.path.exists(TEST_DIR) else []
    if not files:
        print(f"No images found in {TEST_DIR}")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    for file_name in files:
        predict_image(os.path.join(TEST_DIR, file_name), model, scaler)
    print(f"\nFigures saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
