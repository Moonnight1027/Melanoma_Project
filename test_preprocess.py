import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Configuration & Matplotlib Setup
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
SAMPLE_DIR = "./sample_images"
RESULT_DIR = "./result"
IMAGE_NAMES = ["ISIC_0528044.jpg", "ISIC_0080817.jpg", "ISIC_0522926.jpg"]

def mask_circular_fov(image):
    """
    Mask bright/white background corners to absolute black.
    Replaces the flawed 'crop' logic with a robust 'masking' logic.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Isolate very bright pixels (> 240)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological close to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest contour (the skin FOV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask: turn outside region into pure black
        return cv2.bitwise_and(image, image, mask=mask)
    return image

def remove_hair_dullrazor(image):
    """
    DullRazor algorithm: Detect hair via Black-Hat, remove via Inpainting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Black-Hat transform to isolate dark, thin structures
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. Create binary mask for hair
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # 3. Telea inpainting to fill hair pixels
    inpainted_img = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

def advanced_preprocess(image):
    """Full preprocessing pipeline for visualization."""
    masked = mask_circular_fov(image)
    cleaned = remove_hair_dullrazor(masked)
    return cv2.resize(cleaned, (256, 256))

def main():
    """
    Generate a side-by-side visual comparison of original vs processed images.
    """
    print("[INFO] Generating Preprocessing Visualization...")
    plt.figure(figsize=(12, 12))
    
    for idx, img_name in enumerate(IMAGE_NAMES):
        img_path = os.path.join(SAMPLE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
            
        # Read and convert BGR to RGB for matplotlib
        original_img = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Execute preprocessing
        processed_img = advanced_preprocess(original_img)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Plot Original
        plt.subplot(3, 2, idx * 2 + 1)
        plt.title(f"Original: {img_name}")
        plt.imshow(original_rgb)
        plt.axis('off')
        
        # Plot Processed
        plt.subplot(3, 2, idx * 2 + 2)
        plt.title("Processed (FOV Mask + DullRazor)")
        plt.imshow(processed_rgb)
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, "preprocessing_demo_v2.png")
    plt.savefig(save_path, dpi=150) # Increased DPI for sharper presentation
    print(f"[SUCCESS] Visualization saved to '{save_path}'")

if __name__ == "__main__":
    main()