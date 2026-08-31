import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = "./model/resnet18_melanoma.pth"
CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def predict_single_image_dl(image_path):
    print(f"\n[Deep Learning] Analyzing: {image_path}")
    
    # ==========================================
    # 2. Image Preprocessing (Must match training exactly)
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(image_path):
        print("Error: Image not found!")
        return
        
    # Read image using PIL (Torchvision standard)
    img = Image.open(image_path).convert('RGB')
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # ==========================================
    # 3. Load Model
    # ==========================================
    try:
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        # Load weights and map to current device
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        
        # Set to evaluation mode (turns off dropout/batchnorm updates)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # ==========================================
    # 4. Predict
    # ==========================================
    with torch.no_grad():
        outputs = model(img_tensor)
        # Convert raw logits to probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_label = CATEGORIES[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    print(f"DL Result: {predicted_label} (Confidence: {conf_score:.2f}%)")
    
    # ==========================================
    # 5. Visualize and Save
    # ==========================================
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"DL Model: {predicted_label}\nConfidence: {conf_score:.2f}%")
    plt.axis('off')
    
    os.makedirs("./result", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"./result/dl_pred_{base_name}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction saved to {save_path}")

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
                predict_single_image_dl(target_path)
                found = True
                
        if not found:
            print(f"No images found in '{test_dir}'.")