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

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

CATEGORIES = ["Benign", "Malignant"]
FEATURE_NAMES = [
    "HSV_H_Mean", "HSV_H_Std", "HSV_S_Mean", "HSV_S_Std", "HSV_V_Mean", "HSV_V_Std",
    "GLCM_Contrast", "GLCM_Dissim", "GLCM_Homogen", "GLCM_Energy", "GLCM_Correl",
    "Shape_Aspect", "Shape_Extent", "Shape_Solidity"
]

def main():
    print("Loading physical splits data...")
    X_train, y_train = np.load('./data/X_train.npy'), np.load('./data/y_train.npy')
    X_val, y_val = np.load('./data/X_val.npy'), np.load('./data/y_val.npy')
    X_test, y_test = np.load('./data/X_test.npy'), np.load('./data/y_test.npy')
    
    print(f"Data Distribution -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter Tuning using XGBoost
    print("\nStarting Hyperparameter Tuning for XGBoost...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    base_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(estimator=base_xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")

    # Evaluate Test Set
    y_test_pred = best_xgb.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[Test Set Benchmark] Accuracy: {test_acc * 100:.2f}%\n")
    print(classification_report(y_test, y_test_pred, target_names=CATEGORIES))

    # Calculate ROC and AUC
    y_test_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    print(f"XGBoost AUC Score: {roc_auc:.4f}")

    os.makedirs("./result", exist_ok=True)

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'XGBoost Confusion Matrix (Acc: {test_acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./result/benchmark_xgb_matrix.png')

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
    
    # Plot 3: SHAP Feature Importance
    print("\nGenerating SHAP Summary Plot...")
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=FEATURE_NAMES, show=False)
    plt.title('SHAP Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('./result/shap_summary.png')
    print("SHAP Summary saved to './result/shap_summary.png'")

    # Save Model
    os.makedirs("./model", exist_ok=True)
    joblib.dump(best_xgb, './model/melanoma_model.pkl')
    joblib.dump(scaler, './model/feature_scaler.pkl')
    print("\nBest XGBoost Model and Scaler saved successfully!")

if __name__ == "__main__":
    main()