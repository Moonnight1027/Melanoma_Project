import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# System configuration
CSV_PATH = "./train.csv"
IMG_SOURCE_DIR = "./train/"
DATASET_DIR = "./dataset"

def mask_circular_fov(image):
    """
    Mask bright/white background corners (often artifacts of dermoscopes) to pure black.
    This prevents CNNs from learning 'white corners' as a shortcut feature.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Isolate very bright pixels (> 240) which usually represent the background
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological close to bridge small gaps in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest continuous non-white region (the actual skin FOV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Create a black mask and fill the skin area with white (255)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        
        # Apply the mask: retain skin pixels, turn everything else to absolute black (0)
        return cv2.bitwise_and(image, image, mask=mask)
    return image

def remove_hair_dullrazor(image):
    """
    Apply DullRazor algorithm to detect and inpaint hair artifacts.
    Crucial for accurate GLCM texture extraction in downstream tasks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Black-Hat transform: Highlights dark, thin, linear structures (hairs) against a lighter background
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. Thresholding: Create a binary mask of the detected hairs
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # 3. Inpainting: Telea algorithm fills the hair pixels using the colors of surrounding healthy skin
    inpainted_img = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

def advanced_preprocess(image):
    """Execute the full pre-processing pipeline."""
    masked = mask_circular_fov(image)
    cleaned = remove_hair_dullrazor(masked)
    # Standardize output resolution to 256x256
    return cv2.resize(cleaned, (256, 256))

def main():
    print("[INFO] Loading dataset index (train.csv)...")
    df = pd.read_csv(CSV_PATH)
    
    # --- 1. Address Class Imbalance (Under-sampling) ---
    df_benign = df[df.target == 0]
    df_malignant = df[df.target == 1]
    
    print(f"[INFO] Original distribution - Benign: {len(df_benign)}, Malignant: {len(df_malignant)}")
    
    # Randomly under-sample benign images to match the exact number of malignant cases (1:1 ratio)
    df_benign_down = resample(df_benign, replace=False, n_samples=len(df_malignant), random_state=42)
    df_balanced = pd.concat([df_benign_down, df_malignant])
    print(f"[INFO] Balanced dataset size: {len(df_balanced)} images.")
    
    # --- 2. Physical Data Splitting (Stratified 70/15/15) ---
    df_train, df_temp = train_test_split(
        df_balanced, test_size=0.3, random_state=42, stratify=df_balanced['target']
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp['target']
    )
    
    splits = {'train': df_train, 'val': df_val, 'test': df_test}
    categories = {0: "benign", 1: "malignant"}
    
    # --- 3. Execute Pre-processing and Disk I/O ---
    for split_name, df_split in splits.items():
        print(f"\n[INFO] Generating '{split_name}' set ({len(df_split)} images)...")
        
        # Ensure directory structure exists
        for cat in categories.values():
            os.makedirs(os.path.join(DATASET_DIR, split_name, cat), exist_ok=True)
            
        for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
            img_name = row['image_name'] + ".jpg"
            target_label = categories[row['target']]
            
            src_path = os.path.join(IMG_SOURCE_DIR, img_name)
            dst_path = os.path.join(DATASET_DIR, split_name, target_label, img_name)
            
            if os.path.exists(src_path):
                img = cv2.imread(src_path)
                if img is not None:
                    processed_img = advanced_preprocess(img)
                    cv2.imwrite(dst_path, processed_img)

    print("\n[SUCCESS] Data preparation (FOV Masking + DullRazor) completed.")

if __name__ == "__main__":
    main()