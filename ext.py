import os

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./data"
CATEGORIES = ["benign", "malignant"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 256
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def segment_lesion(image):
    """Return a binary mask of the lesion (largest dark region inside the skin area)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Ignore the black background added by prepare_data.py
    _, valid_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    skin_pixels = blurred[valid_mask == 255]
    if len(skin_pixels) == 0:
        return np.zeros_like(gray)

    # Otsu threshold computed on skin pixels only
    thresh_val, _ = cv2.threshold(skin_pixels.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, lesion_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(lesion_mask, valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return final_mask


def extract_features(image):
    """14 features: 6 HSV color, 5 GLCM texture, 3 shape."""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(image)
    if cv2.countNonZero(mask) == 0:
        return np.zeros(14)
    inside = mask == 255

    # Color: mean and std of H, S, V inside the lesion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for ch in range(3):
        values = hsv[:, :, ch][inside]
        features += [np.mean(values), np.std(values)]

    # Texture: gray levels shifted to 1..256 so 0 marks "outside the lesion";
    # dropping row/column 0 keeps only pixel pairs inside the lesion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = np.where(inside, gray.astype(np.uint16) + 1, 0).astype(np.uint16)
    glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=257, symmetric=True)
    glcm = glcm[1:, 1:].astype(np.float64)  # graycoprops normalizes the counts
    features += [graycoprops(glcm, prop)[0, 0] for prop in GLCM_PROPS]

    # Shape: aspect ratio, extent, solidity of the lesion contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    _, _, w, h = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    features += [
        w / h if h else 0,
        area / (w * h) if w * h else 0,
        area / hull_area if hull_area else 0,
    ]
    return np.array(features)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in SPLITS:
        print(f"\nExtracting features: {split}")
        X, y, ids = [], [], []
        for label, category in enumerate(CATEGORIES):
            category_dir = os.path.join(DATASET_DIR, split, category)
            if not os.path.exists(category_dir):
                continue
            files = [f for f in os.listdir(category_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for file_name in tqdm(files, desc=category):
                img = cv2.imread(os.path.join(category_dir, file_name))
                if img is None:
                    continue
                X.append(extract_features(img))
                y.append(label)
                ids.append(os.path.splitext(file_name)[0])

        np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), np.array(X))
        np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), np.array(y))
        # Image names, used by train.py to look up patient IDs
        np.save(os.path.join(OUTPUT_DIR, f"ids_{split}.npy"), np.array(ids))

    print("\nDone.")


if __name__ == "__main__":
    main()
