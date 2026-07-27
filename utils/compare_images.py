import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def compare_images():
    img1 = np.array(Image.open("extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1/clamped_validation_49999.png").convert("RGB"))
    img2 = np.array(Image.open("extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1/clamped_validation_5000.png").convert("RGB"))
    
    if img1.shape != img2.shape:
        print(f"Shape mismatch! img1: {img1.shape}, img2: {img2.shape}")
        return
        
    diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
    print(f"Mean pixel difference: {diff}")
    
    # Calculate difference for specific sections (top half vs bottom half etc)
    diff_top = np.mean(np.abs(img1[:img1.shape[0]//2].astype(float) - img2[:img2.shape[0]//2].astype(float)))
    print(f"Mean pixel difference (top half): {diff_top}")

if __name__ == "__main__":
    compare_images()
