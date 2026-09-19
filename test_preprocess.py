import os
import glob
import cv2
import matplotlib.pyplot as plt

from prepare_data import advanced_preprocess

# ==========================================
# Configuration & Matplotlib Setup
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
SAMPLE_DIR = "./sample_images"
RESULT_DIR = "./result"

def main():
    """
    Generate a side-by-side visual comparison of original vs processed images.
    """
    print("[INFO] Generating Preprocessing Visualization...")
    image_paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    if not image_paths:
        print(f"[WARNING] No images found in '{SAMPLE_DIR}'")
        return
    rows = len(image_paths)
    plt.figure(figsize=(12, 4 * rows))

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        # Read and convert BGR to RGB for matplotlib
        original_img = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Execute preprocessing
        processed_img = advanced_preprocess(original_img)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Plot Original
        plt.subplot(rows, 2, idx * 2 + 1)
        plt.title(f"Original: {img_name}")
        plt.imshow(original_rgb)
        plt.axis('off')
        
        # Plot Processed
        plt.subplot(rows, 2, idx * 2 + 2)
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