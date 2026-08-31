import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# Configuration
DATASET_DIR = "./dataset"
CATEGORIES = ["benign", "malignant"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 256

def segment_lesion(image):
    """Extract lesion by applying Otsu thresholding strictly on non-black valid pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 1. Identify valid skin region (ignore black background from pre-processing)
    _, valid_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    # 2. Extract valid skin pixels
    skin_pixels = blurred[valid_mask == 255]
    if len(skin_pixels) == 0:
        return np.zeros_like(gray)
        
    # 3. Apply Otsu only on valid skin pixels
    skin_pixels_2d = skin_pixels.reshape(-1, 1)
    thresh_val, _ = cv2.threshold(skin_pixels_2d, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply dynamic threshold to the blurred image and mask it with valid region
    _, binary_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(binary_mask, valid_mask)
    
    # 4. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 5. Extract largest contour
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
        
    return final_mask

def extract_features(image):
    """Extract HSV, GLCM, and Shape features."""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(image)
    
    if cv2.countNonZero(mask) == 0:
        return np.zeros(14)

    features = []
    
    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_val = hsv[:, :, 0][mask == 255]
    s_val = hsv[:, :, 1][mask == 255]
    v_val = hsv[:, :, 2][mask == 255]
    features.extend([np.mean(h_val), np.std(h_val), np.mean(s_val), np.std(s_val), np.mean(v_val), np.std(v_val)])
    
    # GLCM
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.bitwise_and(gray, gray, mask=mask)
    glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features.extend([
        graycoprops(glcm, 'contrast')[0, 0], graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0], graycoprops(glcm, 'energy')[0, 0], graycoprops(glcm, 'correlation')[0, 0]
    ])

    # Shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        features.extend([aspect_ratio, extent, solidity])
    else:
        features.extend([0, 0, 0])
        
    return np.array(features)

def main():
    os.makedirs("./data", exist_ok=True)
    
    # Process each physical split separately
    for split in SPLITS:
        print(f"\nExtracting features for [{split}] set...")
        X, y = [], []
        split_dir = os.path.join(DATASET_DIR, split)
        
        for label_idx, category in enumerate(CATEGORIES):
            category_dir = os.path.join(split_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f" -> Category [{category}] ({len(image_files)} images)")
            
            for file_name in tqdm(image_files):
                img = cv2.imread(os.path.join(category_dir, file_name))
                if img is not None:
                    X.append(extract_features(img))
                    y.append(label_idx)

        # Save separately for train/val/test
        np.save(f'./data/X_{split}.npy', np.array(X))
        np.save(f'./data/y_{split}.npy', np.array(y))
        
    print("\nAll features extracted and saved successfully.")
    
if __name__ == "__main__":
    main()