"""Run the full training pipeline in order and stop at the first failure."""
import subprocess
import sys

STEPS = [
    "prepare_data.py",  # balance, clean and split the images
    "ext.py",           # handcrafted features
    "train.py",         # XGBoost + SHAP
    "train_dl.py",      # ResNet18
]


def main():
    for script in STEPS:
        print(f"\n===== {script} =====")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll steps finished. Run predict.py or predict_dl.py for inference.")


if __name__ == "__main__":
    main()
