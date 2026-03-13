import os

# Original logic from helper3dg.py
helper3dg_path = "/mnt/c/Users/kodeb/OneDrive/Desktop/vicks/Infinite-Simul/sim-animate-environment/four_d_gaussian/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/thirdparty/gaussian_splatting/helper3dg.py"
_CUT3R_ROOT = os.path.join(os.path.dirname(helper3dg_path), "..", "..", "..", "..", "..", "..", "CUT3R")
abs_cut3r_root = os.path.abspath(_CUT3R_ROOT)

print(f"Computed CUT3R Root: {abs_cut3r_root}")
print(f"Exists: {os.path.exists(abs_cut3r_root)}")
print(f"Add ckpt path exists: {os.path.exists(os.path.join(abs_cut3r_root, 'add_ckpt_path.py'))}")
