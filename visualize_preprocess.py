"""Save a before/after figure of the preprocessing for every image in sample_images/."""
import glob
import os

import cv2
import matplotlib.pyplot as plt

from prepare_data import advanced_preprocess

SAMPLE_DIR = "./sample_images"
RESULT_DIR = "./result"


def main():
    paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    if not paths:
        print(f"No images found in {SAMPLE_DIR}")
        return

    rows = len(paths)
    plt.figure(figsize=(12, 4 * rows))
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        panels = [
            (f"Original: {os.path.basename(path)}", img),
            ("Processed (FOV Mask + DullRazor)", advanced_preprocess(img)),
        ]
        for j, (title, panel) in enumerate(panels):
            plt.subplot(rows, 2, 2 * i + j + 1)
            plt.title(title)
            plt.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            plt.axis("off")

    plt.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, "preprocessing_demo.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
