import numpy as np
import os

path = "extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1/best_params.npy"
d = np.load(path, allow_pickle=True).item()
d_new = {k: np.array(v) for k, v in d.items()}
np.save("extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1/best_params_np.npy", d_new)
print("Conversion successful.")
