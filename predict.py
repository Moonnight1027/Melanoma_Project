import os
import cv2
import joblib
import matplotlib.pyplot as plt

# Reuse the exact training-time preprocessing and feature extraction
from prepare_data import advanced_preprocess
from ext import extract_features, segment_lesion, IMG_SIZE

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]

def predict_single_image(image_path):
    print(f"\n[Machine Learning] Analyzing: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Same steps as prepare_data.py + ext.py: FOV mask, DullRazor, resize, features
    resized_img = cv2.resize(advanced_preprocess(img), (IMG_SIZE, IMG_SIZE))
    mask = segment_lesion(resized_img)
    features = extract_features(resized_img).reshape(1, -1)

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
    
    print(f"XGBoost Result: {predicted_label} (Confidence: {confidence:.2f}%)")

    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Preprocessed (FOV + DullRazor)\nDiag: {predicted_label}")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Region of Interest Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    os.makedirs("./result", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"./result/xgb_pred_{base_name}.png"
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
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