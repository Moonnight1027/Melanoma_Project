import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# System configuration
DATASET_DIR = "./dataset"
CATEGORIES = ["benign", "malignant"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 256

def segment_lesion(image):
    """
    Segment the lesion area using Otsu's thresholding.
    Strictly ignores the pre-processed black background to avoid threshold bias.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 1. Filter out pure black background (added during prepare_data.py)
    _, valid_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    # 2. Extract only valid skin pixels for threshold calculation
    skin_pixels = blurred[valid_mask == 255]
    if len(skin_pixels) == 0:
        return np.zeros_like(gray)
        
    # 3. Dynamic Otsu thresholding on valid pixels only
    skin_pixels_2d = skin_pixels.reshape(-1, 1)
    thresh_val, _ = cv2.threshold(skin_pixels_2d, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Apply threshold and mask with valid region
    _, binary_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(binary_mask, valid_mask)
    
    # 5. Morphological smoothing (remove noise and fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 6. Keep only the largest contour (the main lesion)
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
        
    return final_mask

def extract_features(image):
    """
    Extract 14 handcrafted features: 6 HSV (Color), 5 GLCM (Texture), 3 Shape.
    """
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(image)
    
    # Return zero vector if no lesion is detected
    if cv2.countNonZero(mask) == 0:
        return np.zeros(14)

    features = []
    
    # --- Feature Set 1: Color (HSV) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_val = hsv[:, :, 0][mask == 255]
    s_val = hsv[:, :, 1][mask == 255]
    v_val = hsv[:, :, 2][mask == 255]
    features.extend([
        np.mean(h_val), np.std(h_val),
        np.mean(s_val), np.std(s_val),
        np.mean(v_val), np.std(v_val)
    ])
    
    # --- Feature Set 2: Texture (GLCM) ---
    # Shift lesion gray levels to 1..256 and leave 0 for "outside the lesion", then drop
    # every pair that touches level 0 so the background does not dominate the GLCM.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = np.where(mask == 255, gray.astype(np.uint16) + 1, 0).astype(np.uint16)
    glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=257, symmetric=True)
    glcm = glcm[1:, 1:].astype(np.float64)  # graycoprops normalizes the counts itself
    features.extend([
        graycoprops(glcm, 'contrast')[0, 0], 
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0], 
        graycoprops(glcm, 'energy')[0, 0], 
        graycoprops(glcm, 'correlation')[0, 0]
    ])

    # --- Feature Set 3: Shape ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        aspect_ratio = float(w) / h if h != 0 else 0
        extent = float(area) / (w * h) if (w * h) != 0 else 0
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        features.extend([aspect_ratio, extent, solidity])
    else:
        features.extend([0, 0, 0])
        
    return np.array(features)

def main():
    """
    Process dataset iteratively and generate feature matrices (.npy).
    """
    os.makedirs("./data", exist_ok=True)
    
    for split in SPLITS:
        print(f"\n[INFO] Extracting features for '{split}' set...")
        X, y, ids = [], [], []
        split_dir = os.path.join(DATASET_DIR, split)
        
        for label_idx, category in enumerate(CATEGORIES):
            category_dir = os.path.join(split_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_name in tqdm(image_files, desc=f"Processing {category}"):
                img_path = os.path.join(category_dir, file_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    X.append(extract_features(img))
                    y.append(label_idx)
                    ids.append(os.path.splitext(file_name)[0])

        # Save extracted features as NumPy arrays
        np.save(f'./data/X_{split}.npy', np.array(X))
        np.save(f'./data/y_{split}.npy', np.array(y))
        np.save(f'./data/ids_{split}.npy', np.array(ids))  # image names, used by train.py for patient-grouped CV
        
    print("\n[SUCCESS] Feature extraction completed.")
    
if __name__ == "__main__":
    main()