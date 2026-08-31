import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def crop_circular_fov(image):
    """移除多餘的白色背景邊框，聚焦中央視野"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找出非純白的區域 (白色通常亮度 > 240)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 形態學閉運算，把散落的皮膚區塊連起來
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 尋找最大輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # 如果找到的輪廓佔據了整張圖片的極大部分(>95%)，代表沒有明顯白邊，不裁切
        img_area = image.shape[0] * image.shape[1]
        if (w * h) / img_area > 0.95:
            return image
            
        return image[y:y+h, x:x+w]
    return image

def remove_hair_dullrazor(image):
    """DullRazor 演算法：移除毛髮並修補背景"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 黑帽運算 (Black-Hat)：強化深色、細長的毛髮結構
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. 建立毛髮遮罩 (Mask)
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # 3. 影像修補 (Inpainting)：用周圍膚色填補毛髮位置
    inpainted_img = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return inpainted_img

def advanced_preprocess(image):
    """整合前處理管線"""
    # Step 1: 裁切白邊
    cropped = crop_circular_fov(image)
    # Step 2: 移除毛髮
    cleaned = remove_hair_dullrazor(cropped)
    # Step 3: 統一縮放回 256x256，確保模型輸入一致
    final_img = cv2.resize(cleaned, (256, 256))
    return final_img

def main():
    sample_dir = "./sample_images"
    image_names = ["ISIC_0528044.jpg", "ISIC_0080817.jpg", "ISIC_0522926.jpg"]
    
    plt.figure(figsize=(12, 12))
    
    for idx, img_name in enumerate(image_names):
        img_path = os.path.join(sample_dir, img_name)
        if not os.path.exists(img_path):
            print(f"找不到檔案: {img_path}")
            continue
            
        # 讀取圖片並轉為 RGB 以供 matplotlib 顯示
        original_img = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 執行進階前處理
        processed_img = advanced_preprocess(original_img)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 繪圖
        plt.subplot(3, 2, idx * 2 + 1)
        plt.title(f"Original: {img_name}")
        plt.imshow(original_rgb)
        plt.axis('off')
        
        plt.subplot(3, 2, idx * 2 + 2)
        plt.title(f"Processed (Crop + DullRazor)")
        plt.imshow(processed_rgb)
        plt.axis('off')

    plt.tight_layout()
    os.makedirs("./result", exist_ok=True)
    plt.savefig("./result/preprocessing_demo.png")
    print("對比圖已儲存至 ./result/preprocessing_demo.png")

if __name__ == "__main__":
    main()