import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
CSV_PATH = "./train.csv"
IMG_SOURCE_DIR = "./train/"
DATASET_DIR = "./dataset"

def mask_circular_fov(image):
    """Mask white background corners to pure black."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # White corners are usually very bright (> 240)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Connect slightly disconnected regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest contour (the actual skin area)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Create a black mask and fill the skin area with white
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask: keep skin area, turn outside into black
        return cv2.bitwise_and(image, image, mask=mask)
    return image

def remove_hair_dullrazor(image):
    """Remove hair using DullRazor algorithm."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Black-Hat transform to find dark, thin hair structures
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Create hair mask
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Inpaint to fill the hair pixels with surrounding skin color
    inpainted_img = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

def advanced_preprocess(image):
    """Apply FOV masking, DullRazor, and resize."""
    masked = mask_circular_fov(image)
    cleaned = remove_hair_dullrazor(masked)
    return cv2.resize(cleaned, (256, 256))

def main():
    print("Reading CSV file...")
    df = pd.read_csv(CSV_PATH)
    
    # Separate classes
    df_benign = df[df.target == 0]
    df_malignant = df[df.target == 1]
    
    # Under-sample benign class
    df_benign_down = resample(df_benign, replace=False, n_samples=len(df_malignant), random_state=42)
    df_balanced = pd.concat([df_benign_down, df_malignant])
    
    # Split dataframe: 70% Train, 30% Temp
    df_train, df_temp = train_test_split(
        df_balanced, test_size=0.3, random_state=42, stratify=df_balanced['target']
    )
    # Split Temp: 50% Val, 50% Test (15% overall each)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp['target']
    )
    
    splits = {'train': df_train, 'val': df_val, 'test': df_test}
    categories = {0: "benign", 1: "malignant"}
    
    # Create directories and process images
    for split_name, df_split in splits.items():
        print(f"\nProcessing [{split_name}] set ({len(df_split)} images)...")
        
        for cat in categories.values():
            os.makedirs(os.path.join(DATASET_DIR, split_name, cat), exist_ok=True)
            
        for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
            img_name = row['image_name'] + ".jpg"
            target_label = categories[row['target']]
            
            src_path = os.path.join(IMG_SOURCE_DIR, img_name)
            dst_path = os.path.join(DATASET_DIR, split_name, target_label, img_name)
            
            if os.path.exists(src_path):
                # Read, apply advanced preprocessing, and save
                img = cv2.imread(src_path)
                if img is not None:
                    processed_img = advanced_preprocess(img)
                    cv2.imwrite(dst_path, processed_img)

    print("\nData preparation with DullRazor & FOV Masking completed!")

if __name__ == "__main__":
    main()