import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import shap

# ==========================================
# 1. System Configuration & Plot Formatting
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
# Explicitly name the 14 features extracted in ext.py for SHAP visualization
FEATURE_NAMES = [
    "Color: HSV_H_Mean", "Color: HSV_H_Std", "Color: HSV_S_Mean", "Color: HSV_S_Std", "Color: HSV_V_Mean", "Color: HSV_V_Std",
    "Texture: GLCM_Contrast", "Texture: GLCM_Dissimilarity", "Texture: GLCM_Homogeneity", "Texture: GLCM_Energy", "Texture: GLCM_Correlation",
    "Shape: Aspect_Ratio", "Shape: Extent", "Shape: Solidity"
]

def main():
    print("\n[INFO] Initializing Machine Learning Pipeline (XGBoost + SHAP)...")
    
    # ==========================================
    # 2. Data Loading & Standardization
    # ==========================================
    print("[INFO] Loading physical splits from NumPy arrays...")
    X_train, y_train = np.load('./data/X_train.npy'), np.load('./data/y_train.npy')
    X_val, y_val = np.load('./data/X_val.npy'), np.load('./data/y_val.npy')
    X_test, y_test = np.load('./data/X_test.npy'), np.load('./data/y_test.npy')
    
    print(f"       -> Distribution: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # Standardize features (mean=0, variance=1)
    # Crucial for models to treat all features (e.g., Hue vs Area) on an equal scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. Model Training & Hyperparameter Tuning
    # ==========================================
    print("\n[INFO] Starting GridSearchCV for XGBoost Hyperparameter Optimization...")
    # Define the search space
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    # Initialize base XGBoost classifier
    base_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    
    # 5-Fold Cross Validation
    grid_search = GridSearchCV(estimator=base_xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"[SUCCESS] Best Parameters Found: {grid_search.best_params_}")

    # ==========================================
    # 4. Final Evaluation (Test Set Metrics)
    # ==========================================
    print("\n[INFO] Executing final benchmark on unseen Test Set...")
    y_test_pred = best_xgb.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n[RESULT] XGBoost Test Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(y_test, y_test_pred, target_names=CATEGORIES))

    # Calculate probabilities for ROC curve (Index 1 is the probability of Malignant)
    y_test_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    print(f"[RESULT] XGBoost AUC Score: {roc_auc:.4f}")

    # ==========================================
    # 5. Visualization Generation
    # ==========================================
    os.makedirs("./result", exist_ok=True)
    print("\n[INFO] Generating analytical plots...")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'XGBoost Confusion Matrix (Accuracy: {test_acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./result/benchmark_xgb_matrix.png')
    print("       -> Saved: Confusion Matrix")

    # Plot 2: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (XGBoost)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./result/benchmark_xgb_roc.png')
    print("       -> Saved: ROC Curve")
    
    # Plot 3: SHAP Feature Importance (Explainable AI)
    print("\n[INFO] Running SHAP TreeExplainer for feature impact analysis...")
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(12, 8)) # Slightly wider for the new feature names
    shap.summary_plot(shap_values, X_test_scaled, feature_names=FEATURE_NAMES, show=False)
    plt.title('SHAP Value Summary (Impact of Features on Model Output)')
    plt.tight_layout()
    plt.savefig('./result/shap_summary.png')
    print("       -> Saved: SHAP Summary Plot")

    # ==========================================
    # 6. Model Persistence
    # ==========================================
    os.makedirs("./model", exist_ok=True)
    joblib.dump(best_xgb, './model/melanoma_model.pkl')
    # The scaler must be saved so the inference engine can standardize new images correctly
    joblib.dump(scaler, './model/feature_scaler.pkl')
    print("\n[SUCCESS] Pipeline Complete. Model and Scaler persisted to disk.")

if __name__ == "__main__":
    main()