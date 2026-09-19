import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Microsoft JhengHei renders the Chinese class names in the plots
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "./data"
RESULT_DIR = "./result"
MODEL_DIR = "./model"
CATEGORIES = ["Benign (良性)", "Malignant (惡性)"]
# Same order as the feature vector built in ext.py
FEATURE_NAMES = [
    "Color: HSV_H_Mean", "Color: HSV_H_Std", "Color: HSV_S_Mean", "Color: HSV_S_Std", "Color: HSV_V_Mean", "Color: HSV_V_Std",
    "Texture: GLCM_Contrast", "Texture: GLCM_Dissimilarity", "Texture: GLCM_Homogeneity", "Texture: GLCM_Energy", "Texture: GLCM_Correlation",
    "Shape: Aspect_Ratio", "Shape: Extent", "Shape: Solidity",
]
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}


def load_split(split):
    X = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    ids = np.load(os.path.join(DATA_DIR, f"ids_{split}.npy"))
    return X, y, ids


def auc_scorer(estimator, X, y):
    """ROC AUC from predict_proba (scoring='roc_auc' does not recognize older XGBoost as a classifier)."""
    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])


def save_confusion_matrix(y_true, y_pred, acc):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f"XGBoost Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "benchmark_xgb_matrix.png"))


def save_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (XGBoost)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "benchmark_xgb_roc.png"))


def save_shap_summary(model, X):
    shap_values = shap.TreeExplainer(model).shap_values(X)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=FEATURE_NAMES, show=False)
    plt.title("SHAP Value Summary (Impact of Features on Model Output)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "shap_summary.png"))


def main():
    X_train, y_train, ids_train = load_split("train")
    X_val, y_val, ids_val = load_split("val")
    X_test, y_test, _ = load_split("test")
    print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # Hyperparameters are chosen by cross-validation, so val is merged into the training data
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    patient_of = pd.read_csv("./train.csv", usecols=["image_name", "patient_id"]).set_index("image_name")["patient_id"]
    groups = patient_of.loc[np.concatenate([ids_train, ids_val])].to_numpy()

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_test_scaled = scaler.transform(X_test)

    # Folds are grouped by patient, like the train/val/test split
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb.XGBClassifier(eval_metric="logloss", random_state=42), PARAM_GRID,
                               cv=cv, scoring=auc_scorer, error_score="raise", n_jobs=-1, verbose=1)
    grid_search.fit(X_fit_scaled, y_fit, groups=groups)
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_} (CV AUC {grid_search.best_score_:.4f})")

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"\nTest accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    print(f"Test AUC: {roc_auc:.4f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    save_confusion_matrix(y_test, y_pred, acc)
    save_roc_curve(fpr, tpr, roc_auc)
    save_shap_summary(model, X_test_scaled)

    # The scaler is needed at inference time as well
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "melanoma_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    print(f"Saved model and scaler to {MODEL_DIR}")


if __name__ == "__main__":
    main()
