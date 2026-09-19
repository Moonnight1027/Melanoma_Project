import os
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from tqdm import tqdm

CSV_PATH = "./train.csv"
IMG_SOURCE_DIR = "./train/"
DATASET_DIR = "./dataset"
IMG_SIZE = 256
CATEGORIES = {0: "benign", 1: "malignant"}

# 20 patient-grouped folds: 3 for test, 3 for val, 14 for train (~15/15/70%)
N_FOLDS = 20
TEST_FOLDS = [0, 1, 2]
VAL_FOLDS = [3, 4, 5]


def mask_circular_fov(image):
    """Black out the bright area outside the dermoscope's circular field of view."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, skin = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)

    # The field of view is the largest non-white region
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(image, image, mask=mask)


def remove_hair_dullrazor(image):
    """DullRazor: find thin dark hairs with a black-hat filter and inpaint them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def advanced_preprocess(image):
    """Preprocessing shared by training and inference."""
    cleaned = remove_hair_dullrazor(mask_circular_fov(image))
    return cv2.resize(cleaned, (IMG_SIZE, IMG_SIZE))


def split_by_patient(df):
    """Split into train/val/test so that each patient appears in only one split."""
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["target"], groups=df["patient_id"])):
        fold[idx] = k
    test = np.isin(fold, TEST_FOLDS)
    val = np.isin(fold, VAL_FOLDS)
    return {"train": df[~test & ~val], "val": df[val], "test": df[test]}


def main():
    df = pd.read_csv(CSV_PATH)
    benign = df[df.target == 0]
    malignant = df[df.target == 1]
    print(f"Original: {len(benign)} benign, {len(malignant)} malignant")

    # Under-sample benign images to a 1:1 ratio
    benign = resample(benign, replace=False, n_samples=len(malignant), random_state=42)
    splits = split_by_patient(pd.concat([benign, malignant]))

    # Rebuild from scratch so images from an earlier split cannot remain
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split_name, df_split in splits.items():
        print(f"\n{split_name}: {len(df_split)} images")
        for cat in CATEGORIES.values():
            os.makedirs(os.path.join(DATASET_DIR, split_name, cat), exist_ok=True)

        for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
            img_name = row["image_name"] + ".jpg"
            img = cv2.imread(os.path.join(IMG_SOURCE_DIR, img_name))
            if img is None:
                continue
            dst = os.path.join(DATASET_DIR, split_name, CATEGORIES[row["target"]], img_name)
            cv2.imwrite(dst, advanced_preprocess(img))

    print("\nDone.")


if __name__ == "__main__":
    main()
