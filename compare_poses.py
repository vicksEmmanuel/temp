#!/usr/bin/env python3
"""
Compare two poses_bounds.npy files (LLFF format)

Format reminder:
    poses_bounds.npy shape: (N, 17)
    columns:
        0-11 : flattened 3x4 camera-to-world matrix (row-major)
        12-14: image height, width, focal length
        15-16: near bound, far bound
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import sys


def load_poses_bounds(path):
    """Load and validate poses_bounds.npy"""
    if not path.endswith('.npy'):
        path = str(path) + '.npy' if not str(path).endswith('.npy') else str(path)
    
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if data.ndim != 2 or data.shape[1] != 17:
        print(f"Unexpected shape for {path}: {data.shape} (expected (N,17))", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {path}: {data.shape[0]} frames")
    return data


def extract_pose_matrices(poses_bounds):
    """Split into c2w (3×4) and hwf + bounds using LLFF interleaved format.
    The 17 values for each frame are:
    [r11, r12, r13, t1, h,
     r21, r22, r23, t2, w,
     r31, r32, r33, t3, f,
     near, far]
    """
    N = poses_bounds.shape[0]
    # Indices for R and t: 0,1,2,3, 5,6,7,8, 10,11,12,13
    poses = poses_bounds[:, :15].reshape(N, 3, 5)
    c2w = poses[:, :3, :4]
    hwf = poses[:, :3, 4]  # [h, w, f]
    bounds = poses_bounds[:, 15:17]  # (N,2) [near, far]
    return c2w, hwf, bounds


def rotation_angle_diff(R1, R2):
    """Geodesic angle difference in degrees between two rotation matrices"""
    try:
        rot1 = Rotation.from_matrix(R1)
        rot2 = Rotation.from_matrix(R2)
        diff = rot1.inv() * rot2
        angle = diff.magnitude() * 180 / np.pi
        return angle
    except:
        return np.nan


def align_poses(t1, t2):
    """
    Very simple Procrustes alignment (Translation + Scale)
    Returns t2 aligned to t1
    """
    # 1. Center both
    mu1 = t1.mean(axis=0)
    mu2 = t2.mean(axis=0)
    
    t1_c = t1 - mu1
    t2_c = t2 - mu2
    
    # 2. Scale alignment
    s1 = np.linalg.norm(t1_c, axis=1).mean()
    s2 = np.linalg.norm(t2_c, axis=1).mean()
    scale = s1 / (s2 + 1e-8)
    
    t2_aligned = t2_c * scale + mu1
    return t2_aligned, scale, mu1 - mu2 * scale


def compare_poses_bounds(file1, file2, rtol=1e-4, atol=1e-3, verbose=True):
    data1 = load_poses_bounds(file1)
    data2 = load_poses_bounds(file2)
    
    if data1.shape[0] != data2.shape[0]:
        print(f"Number of frames differs: {data1.shape[0]} vs {data2.shape[0]}")
        print("→ Comparison stopped (different number of views)")
        return False
    
    N = data1.shape[0]
    c2w1, hwf1, bounds1 = extract_pose_matrices(data1)
    c2w2, hwf2, bounds2 = extract_pose_matrices(data2)
    
    print("\n" + "="*70)
    print(f"Comparing: {file1}  vs  {file2}")
    print("="*70)
    
    # ── Intrinsics / image size ─────────────────────────────────────────────
    focal_diff = np.abs(hwf1[:,2] - hwf2[:,2])
    hw_diff = np.abs(hwf1[:,:2] - hwf2[:,:2])
    
    print("\nIntrinsics summary:")
    print(f"  Focal length diff    mean / max / std : {focal_diff.mean():.4f} / {focal_diff.max():.4f} / {focal_diff.std():.4f}")
    print(f"  Image size diff (h,w) mean / max     : {hw_diff.mean():.4f} / {hw_diff.max():.4f}")
    
    # ── Bounds (near/far) ───────────────────────────────────────────────────
    bounds_diff_near = np.abs(bounds1[:,0] - bounds2[:,0])
    bounds_diff_far  = np.abs(bounds1[:,1] - bounds2[:,1])
    
    print("\nBounds (near/far) differences:")
    print(f"  near: mean / max / std = {bounds_diff_near.mean():.4f} / {bounds_diff_near.max():.4f} / {bounds_diff_near.std():.4f}")
    print(f"  far : mean / max / std = {bounds_diff_far.mean():.4f} / {bounds_diff_far.max():.4f} / {bounds_diff_far.std():.4f}")
    
    # ── Translation differences ─────────────────────────────────────────────
    t1 = c2w1[:, :3, 3]
    t2 = c2w2[:, :3, 3]
    
    trans_diff_raw = np.linalg.norm(t1 - t2, axis=1)
    
    # Align
    t2_aligned, est_scale, est_trans = align_poses(t1, t2)
    trans_diff_aligned = np.linalg.norm(t1 - t2_aligned, axis=1)

    print("\nTranslation (position) differences (RAW):")
    print(f"  mean / median / max / std = {trans_diff_raw.mean():.4f} / {np.median(trans_diff_raw):.4f} / {trans_diff_raw.max():.4f} / {trans_diff_raw.std():.4f}")
    
    print(f"\nTranslation differences (AFTER ALIGNMENT, scale={est_scale:.4f}):")
    print(f"  mean / median / max / std = {trans_diff_aligned.mean():.4f} / {np.median(trans_diff_aligned):.4f} / {trans_diff_aligned.max():.4f} / {trans_diff_aligned.std():.4f}")

    # ── Rotation differences ────────────────────────────────────────────────
    rot_angles = np.array([rotation_angle_diff(c2w1[i,:3,:3], c2w2[i,:3,:3]) for i in range(N)])
    valid = ~np.isnan(rot_angles)
    
    if valid.sum() > 0:
        print("\nRotation differences (degrees):")
        print(f"  mean / median / max / std = {rot_angles[valid].mean():.4f} / {np.median(rot_angles[valid]):.4f} / {rot_angles[valid].max():.4f} / {rot_angles[valid].std():.4f}")
        if (rot_angles[valid] > 45).any():
            print("  → WARNING: some rotations differ by >45°, global alignment probably broken")
    else:
        print("\nRotation: could not compute angles (degenerate matrices?)")
    
    # ── Summary verdict ─────────────────────────────────────────────────────
    print("\n" + "-"*70)
    
    # Use aligned translation for verdict if raw is huge
    effective_trans_error = min(trans_diff_raw.mean(), trans_diff_aligned.mean())
    
    if (focal_diff.max() < 10 and
        effective_trans_error < 0.1 and
        (rot_angles[valid].max() < 5 if valid.sum() > 0 else True)):
        print("→ Poses appear very similar (small differences)")
    elif (focal_diff.max() < 50 and
          effective_trans_error < 0.5 and
          (rot_angles[valid].max() < 15 if valid.sum() > 0 else True)):
        print("→ Poses are roughly aligned, moderate differences")
    else:
        print("→ Significant differences — poses are likely misaligned or from different reconstructions")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two LLFF-style poses_bounds.npy files")
    parser.add_argument("file1", type=str, help="First poses_bounds.npy")
    parser.add_argument("file2", type=str, help="Second poses_bounds.npy")
    parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for intrinsics")
    parser.add_argument("--atol", type=float, default=1e-3, help="absolute tolerance for small differences")
    
    args = parser.parse_args()
    
    compare_poses_bounds(args.file1, args.file2, args.rtol, args.atol)