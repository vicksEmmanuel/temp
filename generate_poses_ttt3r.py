import argparse
import os
import sys
from pathlib import Path
import subprocess
import shutil
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot  # Added for quaternion averaging

# LLFF path for COLMAP-based pose generation
LLFF_PATH = Path("/workspace/sim-animate-environment/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/thirdparty/LLFF")

# Add TTT3R to path
TTT3R_PATH = Path("/workspace/sim-animate-environment/TTT3R")
if str(TTT3R_PATH) not in sys.path:
    sys.path.append(str(TTT3R_PATH))
    sys.path.append(str(TTT3R_PATH / "src"))

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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def average_quaternions(quats):
    # Simple mean and normalize; assumes small variations and same hemisphere
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat

def get_video_dims(video_path):
    import subprocess
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0",
        str(video_path)
    ]
    res = subprocess.check_output(cmd).decode("utf-8").strip()
    w, h = map(int, res.split(","))
    return w, h

def orthogonalize(R):
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        Vt[-1] *= -1
        R_ortho = U @ Vt
    return R_ortho

def procrustes_align(src_centers, dst_centers):
    """Compute similarity transform (scale, rotation, translation) from src to dst.
    
    Uses Procrustes analysis to find: dst ≈ scale * R @ src + t
    
    Args:
        src_centers: (N, 3) source camera centers
        dst_centers: (N, 3) destination camera centers
        
    Returns:
        scale: scalar
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    # Center both sets
    src_mean = src_centers.mean(axis=0)
    dst_mean = dst_centers.mean(axis=0)
    src_c = src_centers - src_mean
    dst_c = dst_centers - dst_mean
    
    # Compute scale
    src_norm = np.sqrt((src_c ** 2).sum())
    dst_norm = np.sqrt((dst_c ** 2).sum())
    scale = dst_norm / (src_norm + 1e-10)
    
    # Compute rotation via SVD
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, d])
    R = Vt.T @ sign_mat @ U.T
    
    # Compute translation
    t = dst_mean - scale * R @ src_mean
    
    return scale, R, t

# LLFF convention matrix: converts OpenCV c2w to LLFF format.
# LLFF swaps rows 0<->1 and negates row 2, matching pose_utils.py line 51:
#   poses = [-u, r, -t] from [r, -u, t]

M_LLFF = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]], dtype=np.float64)

def generate_poses_colmap(video_folder, output_name="poses_bounds.npy"):
    """Generate poses_bounds.npy using COLMAP + LLFF (multi-view SfM).
    
    Extracts one frame per video, runs COLMAP feature extraction/matching/mapping,
    then uses LLFF's pose utilities to produce the standard poses_bounds.npy.
    """
    video_folder = Path(video_folder)
    video_files = sorted(video_folder.glob("cam*.mp4"))
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    print(f"Found {len(video_files)} videos in {video_folder}")
    
    # Step 1: Detect dimensions of the first video and extract 1 frame per video into images/
    target_w, target_h = get_video_dims(video_files[0])
    print(f"Targeting resolution: {target_w}x{target_h} (from {video_files[0].name})")
    
    images_dir = video_folder / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("Extracting frames for COLMAP...")
    for video in video_files:
        output_pattern = images_dir / f"{video.stem}_0001.jpg"
        if not output_pattern.exists():
            subprocess.run([
                "ffmpeg", "-i", str(video),
                "-vf", f"scale={target_w}:{target_h}",
                "-frames:v", "1",
                "-q:v", "2",
                "-loglevel", "error",
                str(images_dir / f"{video.stem}_%04d.jpg")
            ], check=True)
    
    # Step 2: Run LLFF's gen_poses (COLMAP SfM + poses_bounds.npy)
    print("Running COLMAP via LLFF...")
    llff_dir = str(LLFF_PATH)
    sys.path.insert(0, llff_dir)
    
    try:
        from llff.poses.pose_utils import gen_poses
        success = gen_poses(str(video_folder), 'exhaustive_matcher')
        if not success:
            raise RuntimeError("gen_poses returned failure")
    except Exception as e:
        print(f"COLMAP/LLFF failed: {e}")
        print("Falling back to running imgs2poses.py as subprocess...")
        subprocess.run([
            "python3", str(LLFF_PATH / "imgs2poses.py"), str(video_folder)
        ], check=True)
    
    # Step 3: Verify and rename output if needed
    pb_file = video_folder / "poses_bounds.npy"
    if not pb_file.exists():
        raise RuntimeError("poses_bounds.npy was not generated by COLMAP/LLFF")
    
    poses_bounds = np.load(pb_file)
    print(f"✅ COLMAP generated poses_bounds.npy: shape={poses_bounds.shape}")
    
    # If output name differs from default, copy
    if output_name != "poses_bounds.npy":
        output_path = video_folder / output_name
        shutil.copy2(pb_file, output_path)
        print(f"✅ Copied to {output_path}")
    
    # Step 4: Also create colmap_0/sparse/0 structure for downstream pipeline
    colmap_0 = video_folder / "colmap_0"
    sparse_src = video_folder / "sparse" / "0"
    sparse_dst = colmap_0 / "sparse" / "0"
    if sparse_src.exists() and not sparse_dst.exists():
        sparse_dst.mkdir(parents=True, exist_ok=True)
        for f in sparse_src.iterdir():
            shutil.copy2(f, sparse_dst / f.name)
        print(f"✅ Copied sparse model to {sparse_dst}")
    
    # Step 5: Cleanup
    if images_dir.exists():
        shutil.rmtree(images_dir)
    sparse_root = video_folder / "sparse"
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    db_file = video_folder / "database.db"
    if db_file.exists():
        db_file.unlink()
    
    print(f"✅ COLMAP pose generation complete for {video_folder}")

def generate_poses_ttt3r(source_path, output_name="poses_bounds.npy", size=512, model_path="/workspace/sim-animate-environment/pretrain/ttt3r/cut3r_512_dpt_4_64.pth", device="cuda", num_frames_per_video=30):
    source_path = Path(source_path).expanduser().resolve().absolute()
    print(f"DEBUG (TTT3R): Processing source: {source_path}")
    if source_path.is_file():
        video_files = [source_path]
        video_folder = source_path.parent
        is_single_video = True
    else:
        video_folder = source_path
        video_files = sorted(video_folder.glob("cam*.mp4"))
        if not video_files:
            video_files = sorted(video_folder.glob("*.mp4"))
        is_single_video = False

    if not video_files:
        print(f"No video files found at {source_path}")
        return
    print(f"Found {len(video_files)} videos. Single video mode: {is_single_video}", flush=True)

    B = len(video_files)  # Number of cameras
    if is_single_video:
        num_frames = num_frames_per_video
    else:
        num_frames = num_frames_per_video # Use global num_frames if multi-view

    # 1. Get original dimensions and extract all frames
    orig_w, orig_h = get_video_dims(video_files[0])
    print(f"Detected original dimensions: {orig_w}x{orig_h}")
    
    temp_dir = video_folder / "temp_ttt3r_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    raw_img_paths = [[] for _ in range(num_frames)]
    print(f"Extracting {num_frames} frames per video from {B} cameras...", flush=True)
    
    for video_idx, video in enumerate(video_files):
        v_stem = video.stem
        cap = cv2.VideoCapture(str(video))
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        interval = max(1, total_v_frames // num_frames)
        
        for i in range(num_frames):
            frame_idx = i * interval
            output_frame = temp_dir / f"{v_stem}_f{frame_idx:04d}.jpg"
            subprocess.run([
                "ffmpeg", "-i", str(video),
                "-vf", f"select='eq(n,{frame_idx})',scale=1280:-1",
                "-vframes", "1",
                "-q:v", "2",
                "-loglevel", "error",
                "-y",
                str(output_frame)
            ], check=True)
            raw_img_paths[i].append(str(output_frame))
    
    # Get actual resized dimensions from one extracted frame
    temp_img = cv2.imread(raw_img_paths[0][0])
    resized_h, resized_w = temp_img.shape[:2]
    print(f"Resized dimensions for inference: {resized_w}x{resized_h}")
    
    # Interleave: [frame0_cam0, frame0_cam1, ..., frame1_cam0, ...]
    img_paths = [path for frame_list in raw_img_paths for path in frame_list]
    print(f"Total interleaved frames: {len(img_paths)}")

    # TTT3R Imports
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(model_path)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from demo import prepare_input, prepare_output

    # 2. Run TTT3R Recurrent Inference
    print(f"Loading TTT3R model from {model_path}...", flush=True)
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.config.model_update_type = "ttt3r"
    model.eval()

    print(f"Preparing input sequence ({len(img_paths)} total views)...", flush=True)
    img_mask = [True] * len(img_paths)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=size,
        revisit=1,
        update=True,
        reset_interval=10000
    )

    print("Running recurrent inference...")
    with torch.no_grad():
        outputs, state_args = inference_recurrent_lighter(views, model, device)

    # 3. Process outputs
    preds = outputs['pred']
    pts3ds = [p['pts3d_in_self_view'] for p in preds]
    
    # Estimate focal length per camera using all frames
    focals = []
    ws_model = []
    for cam_i in range(B):
        cam_focals = []
        for k in range(num_frames):
            idx = k * B + cam_i
            p3d = pts3ds[idx]
            if p3d.ndim == 3: p3d = p3d.unsqueeze(0)
            C, H, W = p3d.shape[1:4] # pts3ds are [1, H, W, 3] or similar
            # wait, pts3ds[idx] shape is actually [H, W, 3] usually
            if p3d.ndim == 4: # [1, H, W, 3]
                H, W = p3d.shape[1:3]
            else:
                H, W = p3d.shape[0:2]
            ws_model.append(W)
            pp = torch.tensor([W / 2, H / 2], device=device).float().view(1, 2)
            f = estimate_focal_knowing_depth(p3d.to(device), pp.to(device), focal_mode="weiszfeld")
            cam_focals.append(f.item())
        focals.append(np.median(cam_focals))
    
    median_focal_resized = np.median(focals)
    avg_w_model = np.mean(ws_model)
    scale_f = orig_w / avg_w_model
    focal_orig = median_focal_resized * scale_f
    print(f"Estimated focal (model space): {median_focal_resized:.2f} (W={avg_w_model}) -> Original: {focal_orig:.2f}")

    # Extract all poses (c2w in OpenCV convention)
    if 'camera_pose' in preds[0]:
        poses_all = torch.cat([pose_encoding_to_camera(p['camera_pose'].clone()).cpu() for p in preds], dim=0)
    else:
        # Fallback to cumulative relative poses
        world_poses = [torch.eye(4)]
        for p in preds:
            if 'rel_pose' in p:
                world_poses.append(world_poses[-1] @ p['rel_pose'].cpu())
            else:
                world_poses.append(torch.eye(4))
        poses_all = torch.stack(world_poses[:len(preds)])

    # Convert to numpy for easier processing
    c2ws_all_np = poses_all[:, :3, :4].numpy()  # (total_views, 3, 4)

    # Extract camera centers
    centers_all = c2ws_all_np[:, :3, 3]  # (total_views, 3)

    # Align each subsequent time step to the reference using Procrustes
    if not is_single_video:
        ref_centers = centers_all[0:B]
        for k in range(1, num_frames):
            start = k * B
            end = start + B
            src_centers = centers_all[start:end]
            scale, R_align, t_align = procrustes_align(src_centers, ref_centers)

            # Apply alignment to poses in this time step
            for ii in range(B):
                j = start + ii
                old_R = c2ws_all_np[j, :3, :3]
                old_t = c2ws_all_np[j, :3, 3]

                new_t = scale * R_align @ old_t + t_align
                new_R = R_align @ old_R
                new_R = orthogonalize(new_R)

                c2ws_all_np[j, :3, :3] = new_R
                c2ws_all_np[j, :3, 3] = new_t

        # Now average poses per camera across all time steps
        poses_avg = np.zeros((B, 3, 4))
        for cam_i in range(B):
            cam_indices = [f * B + cam_i for f in range(num_frames)]
            cam_Rs = c2ws_all_np[cam_indices, :3, :3]  # (num_frames, 3, 3)
            cam_ts = c2ws_all_np[cam_indices, :3, 3]   # (num_frames, 3)

            # Average translations
            avg_t = np.mean(cam_ts, axis=0)

            # Average rotations via quaternions
            cam_quats = np.array([rotmat2qvec(R) for R in cam_Rs])  # (num_frames, 4)
            avg_quat = average_quaternions(cam_quats)
            avg_R = qvec2rotmat(avg_quat)
            avg_R = orthogonalize(avg_R)

            poses_avg[cam_i, :3, :3] = avg_R
            poses_avg[cam_i, :3, 3] = avg_t
        
        final_poses = poses_avg
        B_out = B
    else:
        # Single video: use all num_frames as separate poses
        final_poses = c2ws_all_np
        B_out = num_frames

    # 4. Calculate Bounds
    print("Estimating near/far bounds...")
    all_bounds = np.zeros((B_out, 2))
    for idx in range(B_out):
        p3d = pts3ds[idx][0].cpu().numpy()
        z_vals = p3d[..., 2].flatten()
        z_pos = z_vals[z_vals > 0]
        if len(z_pos) == 0: z_pos = np.abs(z_vals)
        near = float(np.percentile(z_pos, 0.1))
        far = float(np.percentile(z_pos, 99.9))
        all_bounds[idx] = [near, far]
    bounds = all_bounds

    # 5. Format as LLFF
    print("Converting to LLFF format...")
    llff_poses = []
    for i in range(B_out):
        R_llff = final_poses[i, :3, :3] @ M_LLFF
        t_llff = final_poses[i, :3, 3]
        
        row1 = np.array([R_llff[0,0], R_llff[0,1], R_llff[0,2], t_llff[0], orig_h])
        row2 = np.array([R_llff[1,0], R_llff[1,1], R_llff[1,2], t_llff[1], orig_w])
        row3 = np.array([R_llff[2,0], R_llff[2,1], R_llff[2,2], t_llff[2], focal_orig])
        
        llff_poses.append(np.stack([row1, row2, row3]).flatten())

    llff_poses = np.array(llff_poses)
    save_arr = np.concatenate([llff_poses, bounds], axis=-1)
    
    output_path = video_folder / output_name
    np.save(str(output_path), save_arr)
    print(f"Successfully saved {output_path} with shape {save_arr.shape}")

    # 6. Sparse Model Export
    colmap_0 = video_folder / "colmap_0"
    colmap_0.mkdir(exist_ok=True)
    sparse_path = colmap_0 / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    cx, cy = orig_w / 2.0, orig_h / 2.0
    with open(sparse_path / "cameras.txt", "w") as f:
        f.write("# Camera list: ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {orig_w} {orig_h} {focal_orig} {focal_orig} {cx} {cy}\n")

    with open(sparse_path / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(B_out):
            R_c2w = final_poses[i, :3, :3]
            t_c2w = final_poses[i, :3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qvec = rotmat2qvec(R_w2c)
            # Match the names preprocess.py expects: cam00.png, cam01.png...
            if is_single_video:
                img_name = f"frame_{i:04d}.png"
            else:
                img_name = f"cam{i:02d}.png"
            f.write(f"{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {img_name}\n\n")

    print(f"Exporting 3D points...")
    with open(sparse_path / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        p_idx = 1
        for i in range(0, B_out):
            p3d = pts3ds[i][0].cpu().numpy()
            step = 16
            p3d_sub = p3d[::step, ::step, :].reshape(-1, 3)
            c2w = np.eye(4)
            c2w[:3, :3] = final_poses[i][:3, :3]
            c2w[:3, 3] = final_poses[i][:3, 3]
            pts_world = (p3d_sub @ c2w[:3, :3].T) + c2w[:3, 3]
            for p in pts_world:
                f.write(f"{p_idx} {p[0]} {p[1]} {p[2]} 128 128 128 0\n")
                p_idx += 1

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder of videos (legacy)")
    parser.add_argument("--source", type=str, help="Path to video or folder")
    parser.add_argument("--output", type=str, default="poses_bounds.npy")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_frames_per_video", type=int, default=30)
    parser.add_argument("--use_colmap_poses", action="store_true",
                        help="Use COLMAP+LLFF for pose estimation instead of TTT3R")
    args = parser.parse_args()
    
    source = args.source if args.source else args.folder
    if not source:
        print("Error: --source or --folder required")
    else:
        try:
            if args.use_colmap_poses:
                generate_poses_colmap(source, args.output)
            else:
                generate_poses_ttt3r(source, args.output, args.size, device=args.device,
                                     num_frames_per_video=args.num_frames_per_video)
        except Exception as e:
            print(f"FATAL ERROR in generate_poses_ttt3r: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
