import argparse
import os
import sys
import numpy as np
import torch
import cv2
import shutil
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add PanoDreamer/Depth-Anything-3/src to path
_SCRIPT_DIR = Path(__file__).parent.absolute()
_DA3_PATH = (_SCRIPT_DIR.parent / 'PanoDreamer' / 'Depth-Anything-3' / 'src').resolve()
if str(_DA3_PATH) not in sys.path:
    sys.path.insert(0, str(_DA3_PATH))

try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"[ERROR] Depth-Anything-3 not found: {e}")
    print(f"Path: {sys.path[:3]}") # Show paths
    sys.exit(1)

# Device management
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# LLFF convention matrix
M_LLFF = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]], dtype=np.float64)

def affine_inverse_np(A):
    """Invert a 4x4 or 3x4 rigid transformation matrix."""
    if A.shape == (3, 4):
        A = np.concatenate([A, [[0, 0, 0, 1]]], axis=0)
    R = A[:3, :3]
    T = A[:3, 3:]
    inv_R = R.T
    inv_T = -inv_R @ T
    return np.concatenate([np.concatenate([inv_R, inv_T], axis=1), [[0, 0, 0, 1]]], axis=0)

def get_video_dims(video_path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def generate_poses_da3(source_path, output_name="poses_bounds.npy", checkpoints_dir="../checkpoints", model_repo="depth-anything/DA3NESTED-GIANT-LARGE"):
    source_path = Path(source_path).expanduser().resolve()
    if source_path.is_file():
        video_files = [source_path]
        video_folder = source_path.parent
    else:
        video_folder = source_path
        # Look for cam*.mp4 or all .mp4
        video_files = sorted(video_folder.glob("cam*.mp4"))
        if not video_files:
            video_files = sorted(video_folder.glob("*.mp4"))

    if not video_files:
        print(f"[ERROR] No video files found in {source_path}")
        return

    B = len(video_files)
    print(f"[INFO] Found {B} videos. Processing first frame for pose estimation...")

    # 1. Extract first frame from each video
    temp_dir = video_folder / "temp_da3_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h = get_video_dims(video_files[0])
    img_list = []
    
    for video in video_files:
        cap = cv2.VideoCapture(str(video))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_list.append(img_pil)
        else:
            print(f"[WARNING] Could not read frame from {video}")

    # 2. Initialize and Run DA3
    print(f"[INFO] Initializing DA3 model: {model_repo}")
    model = DepthAnything3.from_pretrained(model_repo, cache_dir=checkpoints_dir)
    model = model.to(DEVICE)
    
    print(f"[INFO] Running DA3 any-view inference on {B} views...")
    with torch.no_grad():
        prediction = model.inference(img_list)

    # 3. Extract and Process outputs
    # prediction.extrinsics: [B, 3, 4] (w2c)
    # prediction.intrinsics: [B, 3, 3]
    # prediction.depth: [B, H, W]
    
    w2cs = prediction.extrinsics # (B, 3, 4)
    ixts = prediction.intrinsics # (B, 3, 3)
    depths = prediction.depth    # (B, H, W)
    
    # We need c2ws in OpenCV for LLFF conversion
    c2ws = []
    for i in range(B):
        # Scale focal to original resolution if DA3 resized them
        # DA3 internal res is usually around 504*N
        # Prediction intrinsics should be scaled back to the 'processed_images' resolution
        # We'll calculate the scale factor relative to original dimensions
        pred_h, pred_w = depths[i].shape
        scale_w = orig_w / pred_w
        scale_h = orig_h / pred_h
        
        # Focal length from intrinsics (assume fx ≈ fy)
        focal = ixts[i][0, 0] * scale_w
        
        # Convert w2c to c2w
        c2w = affine_inverse_np(w2cs[i])[:3, :4]
        
        # Format for LLFF: [R, T, H, W, F]
        # Align to LLFF coordinate system
        R_llff = c2w[:3, :3] @ M_LLFF
        t_llff = c2w[:3, 3]
        
        row1 = np.array([R_llff[0,0], R_llff[0,1], R_llff[0,2], t_llff[0], orig_h])
        row2 = np.array([R_llff[1,0], R_llff[1,1], R_llff[1,2], t_llff[1], orig_w])
        row3 = np.array([R_llff[2,0], R_llff[2,1], R_llff[2,2], t_llff[2], focal])
        
        llff_pose = np.stack([row1, row2, row3]).flatten()
        
        # 4. Near/Far bounds
        z_vals = depths[i].flatten()
        z_pos = z_vals[z_vals > 0]
        if len(z_pos) == 0: z_pos = np.abs(z_vals)
        near = np.percentile(z_pos, 0.1)
        far = np.percentile(z_pos, 99.9)
        
        c2ws.append(np.concatenate([llff_pose, [near, far]]))

    # 5. Save poses_bounds.npy
    save_arr = np.stack(c2ws)
    output_path = video_folder / output_name
    np.save(str(output_path), save_arr)
    print(f"[SUCCESS] Saved {output_path} with shape {save_arr.shape}")

    # 6. Optional: Create COLMAP-style sparse model (cameras.txt, images.txt, points3D.txt)
    colmap_0 = video_folder / "colmap_0"
    colmap_0.mkdir(exist_ok=True)
    sparse_path = colmap_0 / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    # Representative focal/cx/cy (assuming same for all cameras for simplicity, as TTT3R does)
    focal_avg = np.mean([row.reshape(3,5)[2,4] for row in save_arr])
    cx, cy = orig_w / 2.0, orig_h / 2.0
    
    with open(sparse_path / "cameras.txt", "w") as f:
        f.write("# Camera list: ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {orig_w} {orig_h} {focal_avg} {focal_avg} {cx} {cy}\n")

    with open(sparse_path / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(B):
            # We need w2c for COLMAP images.txt
            w2c = w2cs[i]
            # Convert rotation matrix to quaternion
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Simple rotmat to quat (can use scipy or manual if needed)
            def rotmat2qvec(R):
                Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
                K = np.array([
                    [Rxx - Ryy - Rzz, 0, 0, 0],
                    [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                    [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                    [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
                eigvals, eigvecs = np.linalg.eigh(K)
                qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
                if qvec[0] < 0: qvec *= -1
                return qvec

            qvec = rotmat2qvec(R)
            img_name = f"cam{i:02d}.png"
            f.write(f"{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} 1 {img_name}\n\n")

    # Empty points3D.txt for compatibility
    with open(sparse_path / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    shutil.rmtree(temp_dir)
    print(f"[INFO] Sparse model exported to {sparse_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLFF poses and bounds using Depth-Anything-3")
    parser.add_argument("--source", type=str, required=True, help="Folder containing cam*.mp4 videos")
    parser.add_argument("--output", type=str, default="poses_bounds_da3.npy", help="Output filename")
    parser.add_argument("--checkpoints", type=str, default="../checkpoints", help="Checkpoints directory")
    args = parser.parse_args()
    
    generate_poses_da3(args.source, args.output, args.checkpoints)
