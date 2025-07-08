import numpy as np
import os

root_dir = "/Users/doyeonkim/Documents/Project_May/Three_Mode_VaryingMxMy_final/results_coherent"  # update this to your actual root path
#modes = ["qq", "cq", "cc"]
modes = ["qq"]

for mode in modes:
    # Pick any one .npz file from a subfolder of this mode
    mode_path = os.path.join(root_dir, mode)
    subfolders = [f for f in os.listdir(mode_path) if f.startswith("qq")]
    print(subfolders)
    for sub in subfolders:
        subfolder_path = os.path.join(mode_path, sub)
        for fname in os.listdir(subfolder_path):
            if fname.endswith(".npz") and fname.startswith(mode):
                full_path = os.path.join(subfolder_path, fname)
                data = np.load(full_path)
                print(f"{mode.upper()} keys in {fname}:")
                print(data.files)
                break  # only one file needed per mode
        break