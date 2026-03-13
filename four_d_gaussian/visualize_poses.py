#!/usr/bin/env python3
"""
Visualize poses_bounds.npy to assess quality of camera poses.

Generates:
  1. Camera frustum visualization (3D plot saved as PNG)
  2. Numerical summary of camera poses (translations, depth bounds, focal lengths)

Usage:
    python3 visualize_poses.py --source <scene_folder>
    python3 visualize_poses.py --source <scene_folder> --show  # open interactive matplotlib window
"""

import argparse
import numpy as np
from pathlib import Path


def load_poses_bounds(scene_dir):
    """Load poses_bounds.npy and parse into components."""
    pb_path = Path(scene_dir) / "poses_bounds.npy"
    if not pb_path.exists():
        raise FileNotFoundError(f"No poses_bounds.npy found at {pb_path}")
    
    poses_bounds = np.load(str(pb_path))
    n_cams = poses_bounds.shape[0]
    
    # LLFF format: [R(3x3)_col_major | t(3) | hwf(3) | near | far] = 17 values
    poses = poses_bounds[:, :15].reshape(n_cams, 3, 5)
    bounds = poses_bounds[:, 15:]  # near, far
    
    # Extract components
    R = poses[:, :3, :3]  # rotation matrices (3x3)
    t = poses[:, :3, 3]   # translations (3,)
    hwf = poses[:, :3, 4] # height, width, focal
    
    return {
        'n_cams': n_cams,
        'R': R, 't': t, 'hwf': hwf,
        'bounds': bounds,
        'raw': poses_bounds
    }


def print_summary(data):
    """Print numerical summary of poses."""
    print("=" * 60)
    print(f"POSES BOUNDS SUMMARY ({data['n_cams']} cameras)")
    print("=" * 60)
    
    print(f"\nImage dimensions: H={data['hwf'][0, 0]:.0f}, W={data['hwf'][0, 1]:.0f}")
    print(f"Focal lengths: {data['hwf'][:, 2]}")
    print(f"  min={data['hwf'][:, 2].min():.2f}, max={data['hwf'][:, 2].max():.2f}, "
          f"std={data['hwf'][:, 2].std():.2f}")
    
    print(f"\nCamera translations:")
    for i in range(data['n_cams']):
        tx, ty, tz = data['t'][i]
        print(f"  Cam {i}: ({tx:8.4f}, {ty:8.4f}, {tz:8.4f})")
    
    # Translation spread (scene scale indicator)
    t_range = data['t'].max(axis=0) - data['t'].min(axis=0)
    print(f"\n  Translation range: X={t_range[0]:.4f}, Y={t_range[1]:.4f}, Z={t_range[2]:.4f}")
    print(f"  Total spread: {np.linalg.norm(t_range):.4f}")
    
    print(f"\nDepth bounds:")
    for i in range(data['n_cams']):
        near, far = data['bounds'][i]
        print(f"  Cam {i}: near={near:.4f}, far={far:.4f}, ratio={far/max(near, 1e-6):.1f}x")
    
    print(f"\n  Near range: [{data['bounds'][:, 0].min():.4f}, {data['bounds'][:, 0].max():.4f}]")
    print(f"  Far range:  [{data['bounds'][:, 1].min():.4f}, {data['bounds'][:, 1].max():.4f}]")
    
    # Quality checks
    print(f"\n{'=' * 60}")
    print("QUALITY CHECKS")
    print(f"{'=' * 60}")
    
    issues = []
    
    # Check 1: All cameras at same position?
    if np.linalg.norm(t_range) < 1e-4:
        issues.append("⚠️  All cameras at nearly the same position — poses may be incorrect")
    else:
        print("✅ Cameras have spatial spread")
    
    # Check 2: Degenerate depth bounds?
    if (data['bounds'][:, 0] <= 0).any():
        issues.append("⚠️  Some near bounds are <= 0")
    elif (data['bounds'][:, 0] < 1e-4).any():
        issues.append("⚠️  Very small near bounds — potential numerical issues")
    else:
        print("✅ Near bounds are positive")
    
    # Check 3: Near < Far?
    if (data['bounds'][:, 0] >= data['bounds'][:, 1]).any():
        issues.append("⚠️  Some near bounds >= far bounds!")
    else:
        print("✅ Near < Far for all cameras")
    
    # Check 4: Focal length consistency
    focal_std = data['hwf'][:, 2].std()
    focal_mean = data['hwf'][:, 2].mean()
    if focal_std / max(focal_mean, 1e-6) > 0.3:
        issues.append(f"⚠️  Large focal length variation (std/mean = {focal_std/focal_mean:.2f})")
    else:
        print(f"✅ Focal lengths are consistent (std/mean = {focal_std/max(focal_mean, 1e-6):.4f})")
    
    # Check 5: Rotation matrices are proper rotations?
    for i in range(data['n_cams']):
        R = data['R'][i]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.1:
            issues.append(f"⚠️  Cam {i} rotation det={det:.4f} (expected ~1.0)")
    if not any("rotation" in s for s in issues):
        print("✅ All rotation matrices are valid")
    
    # Check 6: Reasonable depth ratio
    depth_ratios = data['bounds'][:, 1] / np.maximum(data['bounds'][:, 0], 1e-6)
    if depth_ratios.max() > 1000:
        issues.append(f"⚠️  Very large depth ratio ({depth_ratios.max():.0f}x) — may cause rendering issues")
    else:
        print(f"✅ Depth ratios are reasonable (max {depth_ratios.max():.1f}x)")
    
    for issue in issues:
        print(issue)
    
    return len(issues) == 0


def visualize_cameras(data, output_path, show=False):
    """Create a 3D visualization of camera frustums."""
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except ImportError:
        print("⚠️  matplotlib not available, skipping visualization")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, data['n_cams']))
    
    # Camera positions
    positions = data['t']
    
    for i in range(data['n_cams']):
        R = data['R'][i]
        t = data['t'][i]
        
        # Draw camera position
        ax.scatter(*t, color=colors[i], s=100, zorder=5)
        ax.text(t[0], t[1], t[2], f'  C{i}', fontsize=8)
        
        # Draw viewing direction (forward = -Z in camera frame, but LLFF convention)
        # In LLFF: row0 = -up, row1 = right, row2 = -forward
        forward = -R[2, :]  # Viewing direction
        scale = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.15
        scale = max(scale, 0.01)
        
        ax.quiver(t[0], t[1], t[2],
                  forward[0], forward[1], forward[2],
                  length=scale, color=colors[i], arrow_length_ratio=0.2, linewidth=2)
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Poses ({data["n_cams"]} cameras)\n'
                 f'Focal: {data["hwf"][0, 2]:.1f}, '
                 f'Near: {data["bounds"][:, 0].mean():.3f}, '
                 f'Far: {data["bounds"][:, 1].mean():.3f}')
    
    # Equal aspect ratio
    positions_range = positions.max(axis=0) - positions.min(axis=0)
    max_range = max(positions_range.max(), 0.01)
    center = positions.mean(axis=0)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize poses_bounds.npy')
    parser.add_argument('--source', required=True, help='Scene folder containing poses_bounds.npy')
    parser.add_argument('--show', action='store_true', help='Show interactive matplotlib window')
    args = parser.parse_args()
    
    data = load_poses_bounds(args.source)
    all_good = print_summary(data)
    
    output_path = Path(args.source) / "camera_poses_visualization.png"
    visualize_cameras(data, output_path, show=args.show)
    
    if all_good:
        print(f"\n🎉 All quality checks passed!")
    else:
        print(f"\n⚠️  Some quality checks failed — review the warnings above")


if __name__ == "__main__":
    main()
