import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
IMG_SIZE = 256

def segment_lesion(image):
    """Extract lesion by first isolating the skin FOV, then applying Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 1. Create Field of View (FOV) mask
    _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    _, light_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    fov_mask = cv2.bitwise_and(dark_mask, light_mask)
    
    # Keep only the largest blob
    contours, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_fov = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_fov, [c], -1, 255, thickness=cv2.FILLED)
        
    # 2. Extract valid skin pixels
    skin_pixels = blurred[clean_fov == 255]
    if len(skin_pixels) == 0:
        return np.zeros_like(gray)
        
    # Compute Otsu threshold only on the skin pixels
    skin_pixels_2d = skin_pixels.reshape(-1, 1)
    thresh_val, _ = cv2.threshold(skin_pixels_2d, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Apply threshold
    _, binary_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(binary_mask, clean_fov)
    
    # 4. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 5. Get largest contour
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
        
    return final_mask

def extract_features(image):
    """Extract features from the resized image and mask."""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(image)
    
    if cv2.countNonZero(mask) == 0:
        return np.zeros(14), mask, image
    
    features = []
    
    # HSV features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_val = hsv[:, :, 0][mask == 255]
    s_val = hsv[:, :, 1][mask == 255]
    v_val = hsv[:, :, 2][mask == 255]
    features.extend([
        np.mean(h_val), np.std(h_val), 
        np.mean(s_val), np.std(s_val), 
        np.mean(v_val), np.std(v_val)
    ])
    
    # GLCM features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.bitwise_and(gray, gray, mask=mask)
    glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features.extend([
        graycoprops(glcm, 'contrast')[0, 0], 
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0], 
        graycoprops(glcm, 'energy')[0, 0], 
        graycoprops(glcm, 'correlation')[0, 0]
    ])
    
    # Shape features
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
        
    return np.array(features), mask, image

def predict_single_image(image_path):
    print(f"\n[Machine Learning] Analyzing: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    features, mask, resized_img = extract_features(img)
    features = features.reshape(1, -1) 

    try:
        model = joblib.load('./model/melanoma_model.pkl')
        scaler = joblib.load('./model/feature_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first.")
        return

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction] * 100
    predicted_label = CATEGORIES[prediction]
    
    print(f"RF Result: {predicted_label} (Confidence: {confidence:.2f}%)")

    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original (Resized)\nDiag: {predicted_label}")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Region of Interest Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    os.makedirs("./result", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"./result/rf_pred_{base_name}.png"
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Prediction saved to '{output_path}'")

if __name__ == "__main__":
    test_dir = "./test"
    
    if not os.path.exists(test_dir):
        print(f"Please create '{test_dir}' folder and add test images.")
    else:
        extensions = [".jpg", ".jpeg", ".png"]
        found = False
        for filename in os.listdir(test_dir):
            if any(filename.lower().endswith(ext) for ext in extensions):
                target_path = os.path.join(test_dir, filename)
                predict_single_image(target_path)
                found = True
                
        if not found:
            print(f"No images found in '{test_dir}'.")