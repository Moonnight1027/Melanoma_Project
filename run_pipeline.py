import subprocess
import sys

# ==========================================
# Configuration: Pipeline Execution Sequence
# ==========================================
PIPELINE_SCRIPTS = [
    "prepare_data.py",  # Step 1: Image cleaning (DullRazor/FOV) & Stratified Splitting
    "ext.py",           # Step 2: Extract 14-dim handcrafted features (ABCD rule)
    "train.py",         # Step 3: Train XGBoost & Generate SHAP explanations
    "train_dl.py"       # Step 4: Train ResNet18 with Dynamic Learning Rate
]

def main():
    """
    Automated executor for the End-to-End Melanoma Classification Pipeline.
    Implements a fail-fast mechanism to halt execution on module errors.
    """
    print("\n[INFO] Starting the Automated Dual-Training Pipeline...\n")
    
    for script in PIPELINE_SCRIPTS:
        print(f"{'='*50}")
        print(f"[EXEC] Running {script}...")
        print(f"{'='*50}")
        
        # Execute the script synchronously using the current Python environment
        result = subprocess.run([sys.executable, script])
        
        # Fail-fast mechanism: halt the pipeline if any module crashes
        if result.returncode != 0:
            print(f"\n[ERROR] '{script}' failed with exit code {result.returncode}.")
            print("[ERROR] Pipeline terminated unexpectedly.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] All pipeline stages completed successfully!")
    print("[INFO] System is ready for inference. Use 'predict.py' or 'predict_dl.py'.\n")

if __name__ == "__main__":
    main()