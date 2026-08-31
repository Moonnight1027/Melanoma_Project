import subprocess
import sys

# Define the sequence of scripts to run
scripts = [
    "prepare_data.py",  # Step 1: Create physical dataset splits (Train/Val/Test)
    "ext.py",           # Step 2: Extract traditional ML features
    "train.py",         # Step 3: Train and evaluate Random Forest (ML)
    "train_dl.py"       # Step 4: Train and evaluate ResNet18 (DL)
]

def main():
    print("Starting the automated dual-training pipeline...\n")
    
    for script in scripts:
        print(f"{'='*50}")
        print(f"Running {script}...")
        print(f"{'='*50}")
        
        # Execute the script using the current Python interpreter
        result = subprocess.run([sys.executable, script])
        
        # Stop the pipeline immediately if a script crashes
        if result.returncode != 0:
            print(f"\n[Error] {script} failed with exit code {result.returncode}.")
            print("Pipeline stopped.")
            sys.exit(result.returncode)

    print("\nAll training steps for both ML and DL completed successfully!")
    print("You can now run 'predict.py' and 'predict_dl.py' for inference testing.")

if __name__ == "__main__":
    main()