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

from scipy.spatial.transform import Rotation as Rot

# Add TTT3R to path
TTT3R_PATH = Path("/workspace/sim-animate-environment/TTT3R")
if str(TTT3R_PATH) not in sys.path:
    sys.path.append(str(TTT3R_PATH))
    sys.path.append(str(TTT3R_PATH / "src"))

def save_as_ply(pts, colors, filename):
    """Save points and colors as a standard PLY file."""
    header = f"""ply
format ascii 1.0
element vertex {len(pts)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p, c in zip(pts, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

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
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat

def orthogonalize(R):
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        Vt[-1] *= -1
        R_ortho = U @ Vt
    return R_ortho

def procrustes_align(src_centers, dst_centers):
    src_mean = src_centers.mean(axis=0)
    dst_mean = dst_centers.mean(axis=0)
    src_c = src_centers - src_mean
    dst_c = dst_centers - dst_mean
    src_norm = np.sqrt((src_c ** 2).sum())
    dst_norm = np.sqrt((dst_c ** 2).sum())
    scale = dst_norm / (src_norm + 1e-10)
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, d])
    R = Vt.T @ sign_mat @ U.T
    t = dst_mean - scale * R @ src_mean
    return scale, R, t

def generate_model_ttt3r(source_path, output_ply="reconstruction.ply", size=800, model_path="/workspace/sim-animate-environment/pretrain/ttt3r/cut3r_512_dpt_4_64.pth", device="cuda", num_frames=30, confidence_threshold=0.5):
    source_path = Path(source_path)
    
    # Identify videos
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

    # 1. Extract frames
    temp_dir = video_folder / "temp_model_frames"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {'Single video' if is_single_video else 'Multi-camera folder'}")
    
    num_cams = len(video_files)
    frames_per_view = num_frames if is_single_video else 10 # 10 frames per camera for multi-view SfM stability
    
    img_paths = []
    cam_indices = [] # Track which camera each frame belongs to
    
    for cam_idx, video in enumerate(video_files):
        cap = cv2.VideoCapture(str(video))
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        interval = max(1, total_v_frames // frames_per_view)
        
        print(f"Extracting {frames_per_view} frames from {video.name}...")
        for i in range(frames_per_view):
            frame_idx = i * interval
            output_frame = temp_dir / f"{video.stem}_f{frame_idx:04d}.jpg"
            subprocess.run([
                "ffmpeg", "-i", str(video),
                "-vf", f"select='eq(n,{frame_idx})',scale={size}:-1",
                "-vframes", "1",
                "-q:v", "2",
                "-loglevel", "error", "-y", str(output_frame)
            ], check=True)
            img_paths.append(str(output_frame))
            cam_indices.append(cam_idx)

    # Reorder img_paths to be interleaved: [f0_c0, f0_c1, ..., f1_c0, f1_c1, ...]
    if not is_single_video:
        interleaved_paths = []
        interleaved_cams = []
        for f in range(frames_per_view):
            for c in range(num_cams):
                idx = c * frames_per_view + f
                interleaved_paths.append(img_paths[idx])
                interleaved_cams.append(c)
        img_paths = interleaved_paths
        cam_indices = interleaved_cams

    # 2. TTT3R Imports & Model Loading
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(model_path)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from demo import prepare_input

    print(f"Loading TTT3R model...")
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.config.model_update_type = "ttt3r"
    model.eval()

    print(f"Preparing input for {len(img_paths)} views...")
    img_mask = [True] * len(img_paths)
    views = prepare_input(img_paths=img_paths, img_mask=img_mask, size=size, revisit=1, update=True)

    print("Running recurrent reconstruction trajectory...")
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, device)

    # 3. Aggregate 3D Points with alignment
    print("Aligning multi-view point clouds...")
    preds = outputs['pred']
    
    # Extract raw world poses (OpenCV convention)
    if 'camera_pose' in preds[0]:
        poses_all = torch.cat([pose_encoding_to_camera(p['camera_pose'].clone()).cpu() for p in preds], dim=0)
    else:
        world_poses = [torch.eye(4)]
        for p in preds:
            if 'rel_pose' in p:
                world_poses.append(world_poses[-1] @ p['rel_pose'].cpu())
            else:
                world_poses.append(torch.eye(4))
        poses_all = torch.stack(world_poses[:len(preds)])

    c2ws_all = poses_all[:, :3, :4].numpy() # (total_views, 3, 4)
    centers_all = c2ws_all[:, :3, 3]
    
    # Global Alignment (Procrustes) for multi-view
    if not is_single_video:
        B = num_cams
        num_timesteps = frames_per_view
        ref_centers = centers_all[0:B]
        
        for k in range(1, num_timesteps):
            start = k * B
            end = start + B
            src_centers = centers_all[start:end]
            scale, R_align, t_align = procrustes_align(src_centers, ref_centers)
            
            for ii in range(B):
                j = start + ii
                old_R = c2ws_all[j, :3, :3]
                old_t = c2ws_all[j, :3, 3]
                new_t = scale * R_align @ old_t + t_align
                new_R = R_align @ old_R
                c2ws_all[j, :3, :3] = orthogonalize(new_R)
                c2ws_all[j, :3, 3] = new_t

    # Averaging Poses per camera
    final_c2ws = []
    if is_single_video:
        final_c2ws = c2ws_all
    else:
        # For multi-view, average across timesteps to get one stable pose per camera
        for cam_i in range(num_cams):
            indices = [f * num_cams + cam_i for f in range(frames_per_view)]
            cam_Rs = c2ws_all[indices, :3, :3]
            cam_ts = c2ws_all[indices, :3, 3]
            
            avg_t = np.mean(cam_ts, axis=0)
            avg_quat = average_quaternions(np.array([rotmat2qvec(R) for R in cam_Rs]))
            avg_R = orthogonalize(qvec2rotmat(avg_quat))
            
            # Use the averaged pose for all frames of this camera (for consistency in point cloud merging)
            # though usually we just want one point cloud per camera or everything in world space.
            m = np.eye(4)
            m[:3, :3] = avg_R
            m[:3, 3] = avg_t
            final_c2ws.append(m)

    # Final aggregation
    all_points = []
    all_colors = []
    
    print("Gathering points ...")
    for i in range(len(preds)):
        pts3d = preds[i]['pts3d_in_self_view'][0].cpu().numpy() # [H, W, 3]
        conf = preds[i]['conf'][0].cpu().numpy() if 'conf' in preds[i] else np.ones(pts3d.shape[:2])
        
        rgb = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (pts3d.shape[1], pts3d.shape[0]))
        
        mask = conf > confidence_threshold
        pts_valid = pts3d[mask]
        colors_valid = rgb[mask]
        
        if is_single_video:
            c2w = c2ws_all[i]
        else:
            # Use the stabilized camera pose for that camera index
            cam_idx = cam_indices[i]
            c2w = final_c2ws[cam_idx]
        
        pts_world = (pts_valid @ c2w[:3, :3].T) + c2w[:3, 3]
        
        # decimate for performance
        step = 8 if is_single_video else 4
        all_points.append(pts_world[::step])
        all_colors.append(colors_valid[::step])

    final_pts = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    print(f"Saving {len(final_pts)} points to {output_ply}...")
    save_as_ply(final_pts, final_colors, output_ply)
    
    shutil.rmtree(temp_dir)
    print("✅ 3D Reconstruction Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to a single .mp4 or a folder of videos")
    parser.add_argument("--output", type=str, default="reconstruction.ply")
    parser.add_argument("--frames", type=int, default=30, help="Frames to use (if single video)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--size", type=int, default=800, help="Inference size")
    args = parser.parse_args()
    
    generate_model_ttt3r(args.source, output_ply=args.output, num_frames=args.frames, confidence_threshold=args.conf, size=args.size)
