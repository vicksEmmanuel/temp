#!/usr/bin/env python3
"""
TTT3R Structure-from-Motion wrapper.
Processes images for a given offset and exports camera poses 
to poses_bounds.npy (LLFF format) for 4D Gaussian Splatting.
"""
import os
import sys
import argparse

# ── TTT3R path setup (MUST happen before any src.dust3r imports) ──
TTT3R_DIR = "/workspace/sim-animate-environment/TTT3R"
TTT3R_SRC = os.path.join(TTT3R_DIR, "src")  # needed for `from dust3r.x` internal imports
DEFAULT_MODEL = "/workspace/sim-animate-environment/pretrain/ttt3r/cut3r_512_dpt_4_64.pth"

sys.path.insert(0, TTT3R_DIR)
sys.path.insert(0, TTT3R_SRC)  # so `import dust3r` works inside the package

def parse_args():
    parser = argparse.ArgumentParser(description="TTT3R SfM wrapper")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to the dataset folder (absolute or relative)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Frame offset to process")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL,
                        help="Path to TTT3R checkpoint")
    return parser.parse_args()

def run_ttt3r_sfm(base_dir, offset, model_path):
    # ── Deferred imports (require add_path_to_dust3r first) ──
    import torch
    import numpy as np
    from src.dust3r.utils.image import load_images
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    device = "cuda"
    img_size = 512

    print(f"[TTT3R-SfM] offset={offset}  base_dir={base_dir}")
    print(f"[TTT3R-SfM] model={model_path}")

    # ── Validate paths ──
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    images_dir = os.path.join(base_dir, f"colmap_{offset}", "input")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(base_dir, f"colmap_{offset}", "images")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(base_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"ERROR: no images dir found for offset {offset} under {base_dir}")
        sys.exit(1)

    img_files = sorted(f for f in os.listdir(images_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if not img_files:
        print(f"ERROR: no images in {images_dir}")
        sys.exit(1)
    img_paths = [os.path.join(images_dir, f) for f in img_files]
    print(f"[TTT3R-SfM] {len(img_paths)} images from {images_dir}")

    # ── Load model ──
    print(f"[TTT3R-SfM] loading model from {model_path}...")
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.config.model_update_type = "cut3r"
    model.eval()
    print("[TTT3R-SfM] model loaded ✓")

    # ── Load images ──
    views_raw = load_images(img_paths, size=img_size)
    print(f"[TTT3R-SfM] images loaded, preparing views…")

    views = []
    for i, v in enumerate(views_raw):
        view = {
            "img": v["img"],
            "ray_map": torch.full(
                (v["img"].shape[0], 6, v["img"].shape[-2], v["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(v["true_shape"]),
            "idx": i,
            "instance": str(i),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)

    print(f"[TTT3R-SfM] views prepared, running inference …")

    # ── Inference ──
    with torch.no_grad():
        outputs, state_args = inference_recurrent_lighter(views, model, device)
    print("[TTT3R-SfM] inference done ✓")

    # ── Extract 3D points and estimate focal ──
    # inference_recurrent_lighter returns a dict: {'views': [...], 'pred': [...]}
    preds = outputs['pred']
    pts3ds = [p['pts3d_in_self_view'] for p in preds]
    pts3ds_other = [p['pts3d_in_other_view'] for p in preds]

    focals = []
    for i in range(len(views)):
        # Ensure pts3d is 4D: (B=1, H, W, 3)
        p3d = pts3ds[i]
        if p3d.ndim == 3:
            p3d = p3d.unsqueeze(0)
        
        # Calculate principal point (pp)
        B, H, W, _ = p3d.shape
        pp = torch.tensor([W / 2, H / 2], device=device).float().view(1, 2)

        f = estimate_focal_knowing_depth(
            p3d.to(device),
            pp.to(device),
            focal_mode="weiszfeld"
        )
        focals.append(f)
    focal = torch.cat(focals).mean()
    print(f"[TTT3R-SfM] focal = {focal.item():.2f}")

    # ── Camera poses ──
    from src.dust3r.utils.camera import pose_encoding_to_camera
    # TTT3R outputs include camera-to-world via pose_encoding or accumulated rel_pose
    if 'camera_pose' in preds[0]:
        poses = torch.cat([pose_encoding_to_camera(p['camera_pose'].clone()).cpu() for p in preds], dim=0)
    else:
        # Build from relative poses
        world_poses = [torch.eye(4, device=device)]
        for p in preds:
            if 'rel_pose' in p:
                world_poses.append(world_poses[-1] @ p['rel_pose'])
            else:
                world_poses.append(torch.eye(4, device=device))
        poses = torch.stack(world_poses[:len(views)]).cpu()

    # ── Get actual image dimensions for intrinsics ──
    # true_shape is (1, 2) = (H_resized, W_resized)
    true_h = int(views[0]['true_shape'][0, 0].item())
    true_w = int(views[0]['true_shape'][0, 1].item())
    focal_val = focal.item()

    # ── Export LLFF poses_bounds.npy ──
    llff_poses = []
    for i in range(len(views)):
        p = poses[i][:3].cpu().numpy()  # 3×4
        hwf = np.array([[true_h], [true_w], [focal_val]])
        llff_poses.append(np.concatenate([p, hwf], axis=1))  # 3×5

    llff_poses = np.stack(llff_poses)  # N×3×5
    bounds = np.tile(np.array([0.1, 100.0]), (len(views), 1))  # N×2
    save_arr = np.concatenate([llff_poses.reshape(-1, 15), bounds], axis=-1)  # N×17

    save_dir = os.path.join(base_dir, f"colmap_{offset}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "poses_bounds.npy")
    np.save(save_path, save_arr)
    print(f"[TTT3R-SfM] saved {save_path}  shape={save_arr.shape}")

    # ── Export COLMAP sparse text files ──
    sparse_dir = os.path.join(save_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # Helper: rotation matrix → COLMAP quaternion (w, x, y, z)
    def rotmat2qvec(R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    # cameras.txt — single shared PINHOLE camera
    cx, cy = true_w / 2.0, true_h / 2.0
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {true_w} {true_h} {focal_val} {focal_val} {cx} {cy}\n")

    # images.txt — one image entry per camera (2 lines each: header + empty points2D)
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(len(views)):
            c2w = poses[i].cpu().numpy()  # 4×4
            # Invert to get w2c
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qvec = rotmat2qvec(R_w2c)
            image_id = i + 1
            f.write(f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                    f"{t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {img_files[i]}\n")
            f.write("\n")  # empty points2D line

    # points3D.txt — Export subsampled points from TTT3R predictions
    print(f"[TTT3R-SfM] Subsampling and exporting 3D points...")
    all_points = []
    all_rgbs = []
    
    step = 4  # 512/4 = 128. 128*128 = 16384 points per image.
    for i in range(len(views)):
        p3d = pts3ds[i] # (1, H, W, 3) or (H, W, 3)
        if p3d.ndim == 4:
            p3d = p3d[0]
        
        # Original image for RGB
        img = views[i]["img"][0].permute(1, 2, 0).cpu().numpy()
        
        # Subsample
        p3d_sub = p3d[::step, ::step, :].reshape(-1, 3).cpu().numpy()
        img_sub = img[::step, ::step, :].reshape(-1, 3)
        
        # Transform to world space
        c2w = poses[i].cpu().numpy()
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = p3d_sub @ R.T + t
        
        all_points.append(pts_world)
        # Convert RGB to [0, 255]
        img_rgb = np.clip(img_sub * 255, 0, 255).astype(np.uint8)
        all_rgbs.append(img_rgb)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_rgbs = np.concatenate(all_rgbs, axis=0)

        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for i in range(len(all_points)):
                p = all_points[i]
                c = all_rgbs[i]
                # ID X Y Z R G B ERROR
                f.write(f"{i+1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")

        print(f"[TTT3R-SfM] saved {len(all_points)} 3D points to points3D.txt")
    else:
        print(f"[TTT3R-SfM] WARNING: No 3D points to export!")

    # ── Ensure images are in the 'images' folder for the Gaussians trainer ──
    input_images_dir = images_dir
    output_images_dir = os.path.join(save_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    import shutil
    for img_f in img_files:
        src_img = os.path.join(input_images_dir, img_f)
        dst_img = os.path.join(output_images_dir, img_f)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
    print(f"[TTT3R-SfM] ensured images are in {output_images_dir}")

    print(f"[TTT3R-SfM] saved COLMAP sparse files in {sparse_dir}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_ttt3r_sfm(args.base_dir, args.offset, args.model_path)
    except Exception as e:
        print(f"FATAL ERROR in ttt3r_sfm.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
