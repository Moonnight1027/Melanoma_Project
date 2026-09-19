import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Reuse the training-time preprocessing (FOV mask + DullRazor + 256x256 resize)
from prepare_data import advanced_preprocess

# System configuration
MODEL_PATH = "./model/resnet18_melanoma.pth"
CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def predict_single_image_dl(image_path, model, transform):
    """Run model inference on a single image and save the result."""
    print(f"[INFO] Analyzing: {os.path.basename(image_path)}")
    
    # Read and apply OpenCV preprocessing
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        return
    cleaned_cv_img = advanced_preprocess(cv_img)
    
    # Convert OpenCV (BGR) to PIL (RGB) for PyTorch transforms
    cleaned_rgb = cv2.cvtColor(cleaned_cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cleaned_rgb)
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        # Convert raw logits to probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_label = CATEGORIES[predicted_idx.item()]
    conf_score = confidence.item() * 100
    print(f"       ResNet18 Result: {predicted_label} (Confidence: {conf_score:.2f}%)")
    
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(cleaned_rgb)
    plt.title(f"ResNet18: {predicted_label}\nConfidence: {conf_score:.2f}%")
    plt.axis('off')
    
    os.makedirs("./result", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"./result/dl_pred_{base_name}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Free memory

def main():
    test_dir = "./test"
    if not os.path.exists(test_dir):
        print(f"[WARNING] Folder '{test_dir}' not found.")
        return

    # Load Model (done once outside the loop to save time)
    print("[INFO] Loading ResNet18 model...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode

    # Define PyTorch transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process all images
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_ext)]
    
    if not image_files:
        print(f"[INFO] No valid images found in '{test_dir}'.")
        return
        
    for filename in image_files:
        predict_single_image_dl(os.path.join(test_dir, filename), model, transform)
        
    print("\n[SUCCESS] All predictions saved to './result/'")

if __name__ == "__main__":
    main()