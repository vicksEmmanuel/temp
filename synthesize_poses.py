import os
import json
import numpy as np
import argparse

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

poses_list = [
    "w-31", "s-31", "a-31", "d-31", 
    "up-31", "down-31", "left-31", "right-31",
    "w-15, d-16", "w-15, a-16"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="generated_scene_01")
    parser.add_argument("--num_videos", type=int, default=18)
    parser.add_argument("--frames_per_video", type=int, default=13)
    args = parser.parse_args()

    num_videos = args.num_videos
    frames_per_video = args.frames_per_video
    scene_name = args.scene_name

    width = 1280
    height = 704

    fx_norm = 969.6969696969696 / 1920.0
    fy_norm = 969.6969696969696 / 1080.0
    fx_px = fx_norm * width
    fy_px = fy_norm * height 

    K_px = [
        [fx_px, 0.0, width / 2.0],
        [0.0, fy_px, height / 2.0],
        [0.0, 0.0, 1.0]
    ]

    output_data = {}
    
    # 1. Build a single continuous trajectory
    continuous_trajectory = []
    current_T = np.eye(4)
    
    forward_speed = 0.08
    jaw_speed = np.deg2rad(1.5)
    pitch_speed = np.deg2rad(1.5)

    for i in range(num_videos):
        pose_str = poses_list[i % len(poses_list)]
        
        forward = 0.0
        right = 0.0
        pitch = 0.0
        yaw = 0.0
        
        # Simplified parsing of the dominant motion intent in the string segment
        if "w" in pose_str: forward = forward_speed
        if "s" in pose_str: forward = -forward_speed
        if "a" in pose_str: right = -forward_speed
        if "d" in pose_str: right = forward_speed
        if "up" in pose_str: pitch = pitch_speed
        if "down" in pose_str: pitch = -pitch_speed
        if "left" in pose_str: yaw = -jaw_speed
        if "right" in pose_str: yaw = jaw_speed

        for t in range(frames_per_video):
            R = rot_y(yaw) @ rot_x(pitch)
            current_T[:3, :3] = current_T[:3, :3] @ R
            
            local_t = np.array([right, 0, forward])
            world_t = current_T[:3, :3] @ local_t
            current_T[:3, 3] += world_t
            
            continuous_trajectory.append(current_T.copy())

    # 2. Extract specific absolute segments matching the chronologically captured segments
    poses_bounds_list = []
    
    for frame_offset in range(frames_per_video):
        output_data[str(frame_offset)] = {}
        for i in range(num_videos):
            cam_name = f"cam{i:02d}"
            # Global index along the single continuous video timeline
            global_frame_idx = i * frames_per_video + frame_offset
            c2w = continuous_trajectory[global_frame_idx]
            
            output_data[str(frame_offset)][cam_name] = {
                "extrinsic": c2w.tolist(),
                "K": K_px
            }

            # Only for frame 0, or we can use a per-timestamp poses_bounds if needed
            # but usually SpacetimeGS wants one global poses_bounds.npy in the parent dir
            if frame_offset == 0:
                # LLFF format: [R | T | HWf]
                # We need to construct a 3x5 matrix:
                # [ R11 R12 R13 Tx  H ]
                # [ R21 R22 R23 Ty  W ]
                # [ R31 R32 R33 Tz  f ]
                
                # Note: LLFF orientation is often [right, -up, -back].
                # Our c2w is likely [right, down, forward] or similar.
                # However, for synthetic data, as long as it's consistent, 3DGS usually works.
                # Standard LLFF loader in this repo: poses = poses_bounds[:, :15].reshape(-1, 3, 5)
                
                llff_pose = np.zeros((3, 5))
                llff_pose[:3, :3] = c2w[:3, :3]
                llff_pose[:3, 3] = c2w[:3, 3]
                llff_pose[0, 4] = height
                llff_pose[1, 4] = width
                llff_pose[2, 4] = fx_px
                
                # Add near/far bounds (dummies)
                row = np.concatenate([llff_pose.ravel(), [0.1, 100.0]])
                poses_bounds_list.append(row)

    out_folder = os.path.join("infinite-simul-realtime-4d-gaussian-vgg", "data", scene_name, scene_name)
    os.makedirs(out_folder, exist_ok=True)
    
    # Save NPY
    if poses_bounds_list:
        npy_path = os.path.join(out_folder, "poses_bounds.npy")
        np.save(npy_path, np.array(poses_bounds_list))
        print(f"Successfully synthesized {npy_path}")

    out_path = os.path.join(out_folder, "synthesized_poses.json")
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully synthesized analytical poses to {out_path}")
    print(f"Contains {num_videos} cameras across {frames_per_video} frames.")
