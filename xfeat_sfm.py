
import os
import sys
import torch
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import cv2
from pathlib import Path
import subprocess
import sqlite3

# Add XFeat to path
XFEAT_PATH = "/workspace/sim-animate-environment/accelerated_features"
if XFEAT_PATH not in sys.path:
    sys.path.append(XFEAT_PATH)

from modules.xfeat import XFeat

# Add 3DGS utils to path
UTILS_PATH = "/workspace/sim-animate-environment/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite-simul-spacetime-gaussian/thirdparty/gaussian_splatting/utils"
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from pre_colmap import COLMAPDatabase

def run_xfeat_sfm(image_dir, database_path, output_path, match_type='sequential'):
    """
    image_dir: Path to directory with images (e.g., cam00.png, cam01.png ...)
    database_path: Path to the .db file to create
    output_path: Path to the sparse/0 reconstruction folder
    """
    image_dir = Path(image_dir)
    database_path = Path(database_path)
    output_path = Path(output_path)
    image_files = sorted([f for f in image_dir.glob("*.png") or image_dir.glob("*.jpg") or image_dir.glob("*.jpeg")])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # 1. Initialize XFeat
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xfeat = XFeat().to(device).eval()
    print(f"XFeat initialized on {device}")

    # 2. Create/Clear database
    if os.path.exists(database_path):
        os.remove(database_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    # 3. Add Camera
    # Using SIMPLE_RADIAL model (id 1)
    # Params: [f, cx, cy, k]
    img_sample = cv2.imread(str(image_files[0]))
    h, w = img_sample.shape[:2]
    f = 1.2 * max(h, w) # Rough guess
    cx, cy = w / 2.0, h / 2.0
    camera_id = db.add_camera(model=1, width=w, height=h, params=[f, cx, cy, 0.0])

    # 4. Extract Features
    image_id_map = {} # name -> id
    features_cache = {} # id -> out_dict
    
    print("Extracting XFeat features...")
    for img_path in image_files:
        img_name = img_path.name
        image_id = db.add_image(img_name, camera_id)
        image_id_map[img_name] = image_id
        
        img = cv2.imread(str(img_path))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device) / 255.0
        
        with torch.no_grad():
            out = xfeat.detectAndCompute(img_tensor, top_k=4096)[0]
        
        # Save keypoints (N, 2)
        kpts = out['keypoints'].cpu().numpy()
        db.add_keypoints(image_id, kpts)
        
        features_cache[image_id] = {
            'keypoints': out['keypoints'],
            'descriptors': out['descriptors']
        }
        
    db.commit()

    # 5. Matching
    print(f"Matching features ({match_type})...")
    
    pairs = []
    if match_type == 'sequential':
        for i in range(len(image_files) - 1):
            pairs.append((image_id_map[image_files[i].name], image_id_map[image_files[i+1].name]))
            # Also match with a small lookback
            if i > 0:
                pairs.append((image_id_map[image_files[i-1].name], image_id_map[image_files[i+1].name]))
    else: # exhaustive
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                pairs.append((image_id_map[image_files[i].name], image_id_map[image_files[j].name]))

    for id1, id2 in pairs:
        desc1 = features_cache[id1]['descriptors']
        desc2 = features_cache[id2]['descriptors']
        
        with torch.no_grad():
            idx0, idx1 = xfeat.match(desc1, desc2)
            
        if len(idx0) > 15: # Min matches
            matches = np.stack([idx0.cpu().numpy(), idx1.cpu().numpy()], axis=1).astype(np.uint32)
            db.add_matches(id1, id2, matches)
            # For simplicity, we can also add as two_view_geometry with generic config
            # config=1 (basic) or config=2 (calibrated)
            db.add_two_view_geometry(id1, id2, matches)

    db.commit()
    db.close()
    print("Database preparation complete.")

    # our llff_base is the parent of output_path
    llff_base = output_path.parent
    colmap_output = output_path
    
    # 6. Run COLMAP Mapper
    os.makedirs(colmap_output, exist_ok=True)
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(colmap_output),
        '--Mapper.num_threads', '16',
        '--Mapper.ba_refine_focal_length', '1',
        '--Mapper.ba_refine_extra_params', '1',
    ]
    
    print("Running COLMAP mapper...")
    try:
        subprocess.run(mapper_args, check=True)
        print(f"SfM finished. Results in {colmap_output}")
    except subprocess.CalledProcessError as e:
        print(f"COLMAP mapper failed: {e}")

    # 7. Generate poses_bounds.npy using LLFF utilities
    print("Generating poses_bounds.npy...")
    LLFF_PATH = "/workspace/sim-animate-environment/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite-simul-spacetime-gaussian/thirdparty/LLFF"
    if LLFF_PATH not in sys.path:
        sys.path.append(LLFF_PATH)
    
    try:
        from llff.poses.pose_utils import load_colmap_data, save_poses
        
        print(f"Loading COLMAP data from {llff_base}...")
        
        # load_colmap_data expects the parent of 'sparse'
        # if colmap_output is .../sparse, then it contains '0/*.bin'
        # so llff_base should be the directory containing 'sparse'
        poses, pts3d, perm = load_colmap_data(str(llff_base))
        save_poses(str(llff_base), poses, pts3d, perm)
        print(f"Successfully generated {llff_base}/poses_bounds.npy")
    except Exception as e:
        print(f"Failed to generate poses_bounds.npy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--match_type", type=str, default="sequential")
    args = parser.parse_args()
    
    run_xfeat_sfm(args.image_dir, args.db_path, args.output_path, args.match_type)
