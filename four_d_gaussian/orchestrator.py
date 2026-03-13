import os
import numpy as np
if not hasattr(np, 'bool_'):
    np.bool_ = getattr(np, 'bool', bool)
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import argparse
import subprocess
import shutil
from pathlib import Path
import sys
import torch
import time
import requests
import json

# ── RunwayML SDK ──
try:
    from runwayml import RunwayML
except ImportError:
    print("RunwayML SDK not found. Install it with: pip install runwayml")
    RunwayML = None

# ── RunwayML API Key ──
RUNWAYML_API_KEY = "key_8bf56626283170655f68c12c2b4118f4e9209c949ae4f83fb9b54cb87e4780daa71665edc53b17ef37e0bf44d8dd4c735fb6292bf84347a5604da4f0e761bda1"

# Paths Setup — auto-detect from script location
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent  # four_d_gaussian -> sim-animate-environment
VGG_GAUSSIAN_DIR = WORKSPACE_ROOT / "four_d_gaussian" / "infinite-simul-realtime-4d-gaussian-vgg"
VGG_DATA_DIR = VGG_GAUSSIAN_DIR / "data"
WAN_DIR = WORKSPACE_ROOT / "Wan2.1"
COSMOS_DIR = WORKSPACE_ROOT / "cosmos-predict2.5"
DEFAULT_VACE_MODEL = WORKSPACE_ROOT / "pretrain" / "Wan2.1-VACE-1.3B"
DEFAULT_I2V_MODEL = WORKSPACE_ROOT / "pretrain" / "Wan2.1-I2V-14B-480P"
DEFAULT_COSMOS_MODEL_DIR = WORKSPACE_ROOT / "pretrain" / "Cosmos-2.5-2B"

# Fix for shared library issues
TORCH_LIB_PATH = "/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib"

# Camera-movement prompts for generating viewport variations
CAMERA_PROMPTS = [
    "A slow orbital pan around the scene, keeping the main subject centered while the camera moves in a smooth arc.",
    "A dynamic low-angle dolly shot, moving forward into the scene to emphasize depth and scale.",
    "A smooth high-angle crane shot, slowly descending while tilting up to reveal more of the background.",
    "A cinematic lateral tracking shot, moving from left to right at eye level to follow the action.",
    "A slow zoom-in from a distant perspective, gradually narrowing the field of view on the central elements.",
    "A 360-degree orbital rotation around the environment, capturing all angles in a single smooth motion.",
    "A dramatic Dutch angle tilt combined with a slow pedestal up movement.",
    "A wide-angle sweep from a high corner, panning slowly across the entire landscape.",
    "A handheld-style shaky cam effect, moving forward with subtle organic jitters for realism.",
    "A bird's eye view with a slow clockwise rotation, maintaining a consistent downward perspective.",
    "A ground-level slider shot, moving right to left with a focus on foreground textures.",
    "A slow pull-back dolly shot, starting close and gradually revealing the vastness of the environment.",
    "A diagonal tracking shot, moving from high-left to low-right while keeping the horizon stable.",
    "An over-the-shoulder perspective that slowly orbits the subject to reveal the reverse angle.",
    "A static but atmospheric shot with a slow lens zoom-out, emphasizing the subject's isolation in the space.",
    "A fast-paced slider movement with a slight whip-pan at the end for a dynamic transition feel.",
    "A smooth tilt-down from the sky to the main action, settling into a medium-close shot.",
    "A slow forward push-in with a slight roll left, creating a sense of immersive investigation.",
    "A wide panoramic sweep, rotating 180 degrees from a fixed central position.",
    "A low-angle heroic shot with a slow counter-clockwise wrap-around movement.",
]


def copy_videos_from_folder(scene_name: str, video_folder: str):
    scene_dir = VGG_DATA_DIR / scene_name / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_folder)
    
    print(f"Copying videos from {video_folder} to {scene_dir}...")
    videos = sorted(list(video_path.glob("*.mp4")))
    if not videos:
        print(f"Warning: No .mp4 files found in {video_folder}")
        return
        
    for i, vid in enumerate(videos):
        dest_path = scene_dir / f"cam{i:02d}.mp4"
        shutil.copy2(vid, dest_path)
        print(f"✅ Copied {vid.name} to {dest_path.name}")
    
    # Also copy poses_bounds.npy if it exists
    pb_source = video_path / "poses_bounds.npy"
    if pb_source.exists():
        shutil.copy2(pb_source, scene_dir / "poses_bounds.npy")
        print(f"✅ Copied poses_bounds.npy to {scene_dir}")
    else:
        print(f"ℹ️ No poses_bounds.npy found in {video_folder}")
        
    print("Video copy complete.")


def generate_vace_videos(scene_name: str, base_image: str, base_prompt: str,
                         num_videos: int, vace_model: str, i2v_model: str = None,
                         sample_steps: int = 50, guide_scale: float = 5.0):
    scene_dir = VGG_DATA_DIR / scene_name / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = WORKSPACE_ROOT / "outputs" / scene_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WAN_DIR) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    model_for_step1 = vace_model
    task_for_step1 = "vace-1.3B"
    
    print(f"\n{'='*60}")
    print(f"[WAN2.1] Step 1: Generating base video from image + prompt")
    print(f"[WAN2.1] Model/Task: {task_for_step1} (first-frame conditioned)")
    print(f"[WAN2.1] Image: {base_image}")
    print(f"[WAN2.1] Prompt: {base_prompt[:80]}...")
    print(f"{'='*60}\n")

    base_video_path = outputs_dir / "cam00.mp4"

    if base_video_path.exists():
        print(f"⏩ Step 1: Base video already exists at {base_video_path}. Skipping generation.")
    else:
        vace_prep_dir = outputs_dir / "vace_prep"
        print("[PREP] Creating first-frame source video and mask...")
        subprocess.run([
            "python3", str(WORKSPACE_ROOT / "prepare_vace_source.py"),
            "--image", base_image,
            "--output_dir", str(vace_prep_dir),
        ], check=True)
        src_video_path = str(vace_prep_dir / "src_video.mp4")
        src_mask_path = str(vace_prep_dir / "src_mask.mp4")

        cmd_r2v = [
            "python3", "generate.py",
            "--task", task_for_step1,
            "--size", "832*480",
            "--ckpt_dir", str(model_for_step1),
            "--src_video", src_video_path,
            "--src_mask", src_mask_path,
            "--src_ref_images", base_image,
            "--prompt", base_prompt,
            "--save_file", str(base_video_path),
            "--offload_model", "True",
            "--t5_cpu",
            "--base_seed", "42",
            "--sample_steps", str(sample_steps),
            "--sample_guide_scale", str(guide_scale),
        ]

        try:
            subprocess.run(cmd_r2v, cwd=str(WAN_DIR), env=env, check=True)
            print(f"✅ Base video saved to {base_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating base video: {e}")
            raise e

    dest_cam00 = scene_dir / "cam00.mp4"
    shutil.copy2(str(base_video_path), str(dest_cam00))
    print(f"✅ Copied base video to {dest_cam00}")

    for i in range(1, num_videos):
        cam_prompt_idx = (i - 1) % len(CAMERA_PROMPTS)
        cam_prompt = CAMERA_PROMPTS[cam_prompt_idx]
        full_prompt = f"{base_prompt}. {cam_prompt}"
        seed = 42 + i

        print(f"\n{'='*60}")
        print(f"[VACE] Step 2: Generating cam{i:02d} ({i}/{num_videos-1})")
        print(f"[VACE] Camera: {cam_prompt[:60]}...")
        print(f"{'='*60}\n")

        cam_video_path = outputs_dir / f"cam{i:02d}.mp4"

        if cam_video_path.exists():
            print(f"⏩ cam{i:02d} already exists at {cam_video_path}. Skipping generation.")
        else:
            cmd_v2v = [
                "python3", "generate.py",
                "--task", "vace-1.3B",
                "--size", "832*480",
                "--ckpt_dir", str(vace_model),
                "--src_video", str(base_video_path),
                "--src_ref_images", base_image,
                "--prompt", full_prompt,
                "--save_file", str(cam_video_path),
                "--offload_model", "True",
                "--t5_cpu",
                "--base_seed", str(seed),
                "--sample_steps", str(sample_steps),
                "--sample_guide_scale", str(guide_scale),
            ]

            try:
                torch.cuda.empty_cache()
                subprocess.run(cmd_v2v, cwd=str(WAN_DIR), env=env, check=True)
                print(f"✅ cam{i:02d} saved to {cam_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error generating cam{i:02d}: {e}")
                continue

        dest_path = scene_dir / f"cam{i:02d}.mp4"
        shutil.copy2(str(cam_video_path), str(dest_path))
        print(f"✅ Copied to {dest_path}")

    print(f"\n[ORCHESTRATOR] VACE video generation complete. {num_videos} cameras generated.")


def generate_cosmos_videos(scene_name: str, base_image: str, base_prompt: str,
                           num_videos: int, cosmos_model_dir: str,
                           sample_steps: int = 35, guide_scale: float = 7.0):
    scene_dir = VGG_DATA_DIR / scene_name / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = WORKSPACE_ROOT / "outputs" / scene_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_video_path = outputs_dir / "cam00.mp4"
    if not base_video_path.exists():
        print("[COSMOS] Base video missing. Generating using VACE first...")
        generate_vace_videos(scene_name, base_image, base_prompt, 1, str(DEFAULT_VACE_MODEL))

    import json
    template_path = Path(__file__).parent / "cosmos_inference_template.json"
    if not template_path.exists():
        raise FileNotFoundError("cosmos_inference_template.json not found.")

    with open(template_path, "r") as f:
        template = json.load(f)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(WORKSPACE_ROOT / "mock_te") + ":" +
        str(COSMOS_DIR) + ":" + 
        str(COSMOS_DIR / "packages" / "cosmos-oss") + ":" + 
        str(COSMOS_DIR / "packages" / "cosmos-cuda") + ":" + 
        env.get("PYTHONPATH", "")
    )
    env["PATH"] = "/root/.local/bin:" + env.get("PATH", "")
    env["VERBOSE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    ckpt_path = Path(cosmos_model_dir) / "base" / "post-trained" / "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

    for i in range(1, num_videos):
        cam_prompt_idx = (i - 1) % len(CAMERA_PROMPTS)
        cam_prompt = CAMERA_PROMPTS[cam_prompt_idx]
        full_prompt = f"{base_prompt}. {cam_prompt}"
        seed = 42 + i

        print(f"\n{'='*60}")
        print(f"[COSMOS] Step 2: Generating cam{i:02d} ({i}/{num_videos-1})")
        print(f"[COSMOS] Camera: {cam_prompt[:60]}...")
        print(f"{'='*60}\n")

        cam_video_path = outputs_dir / f"cam{i:02d}.mp4"
        
        if cam_video_path.exists():
            print(f"⏩ Cosmos: cam{i:02d} already exists at {cam_video_path}. Skipping generation.")
        else:
            job_json = template.copy()
            job_json["name"] = f"cam{i:02d}"
            job_json["inference_type"] = "video2world"
            job_json["input_path"] = str(base_video_path)
            job_json["prompt"] = full_prompt
            job_json["seed"] = seed
            job_json["num_steps"] = sample_steps
            job_json["guidance"] = int(min(7, max(1, guide_scale)))
            
            job_file = outputs_dir / f"job_cam{i:02d}.json"
            with open(job_file, "w") as f:
                json.dump(job_json, f)

            cmd_cosmos = [
                "python3", "examples/inference.py",
                "--output-dir", str(outputs_dir),
                "--checkpoint-path", str(ckpt_path),
                "--model", "2B/post-trained",
                "--input-files", str(job_file),
                "--disable-guardrails"
            ]

            try:
                torch.cuda.empty_cache()
                subprocess.run(cmd_cosmos, cwd=str(COSMOS_DIR), env=env, check=True)
                print(f"✅ Cosmos: cam{i:02d} generation complete.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error generating cam{i:02d} with Cosmos: {e}")
                continue

        if cam_video_path.exists():
            dest_path = scene_dir / f"cam{i:02d}.mp4"
            shutil.copy2(str(cam_video_path), str(dest_path))
            print(f"✅ Copied {cam_video_path.name} to {dest_path}")
        else:
            print(f"⚠️ Cosmos: cam{i:02d} file missing at {cam_video_path}.")

    print(f"\n[ORCHESTRATOR] Cosmos video generation complete. {num_videos} cameras generated.")


def generate_runwayml_videos(scene_name: str, base_prompt: str, num_videos: int,
                             sample_steps: int = 20, guide_scale: float = 8.0):
    if RunwayML is None:
        print("Cannot use RunwayML: SDK not installed. Run: pip install runwayml")
        return

    scene_dir = VGG_DATA_DIR / scene_name / scene_name
    outputs_dir = WORKSPACE_ROOT / "outputs" / scene_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_video_path = outputs_dir / "cam00.mp4"
    if not base_video_path.exists():
        raise FileNotFoundError("Base video cam00.mp4 not found - run VACE first or provide --video_folder")

    client = RunwayML(api_key=RUNWAYML_API_KEY)

    compressed_base = outputs_dir / "cam00_compressed_runway.mp4"
    if not compressed_base.exists():
        print(f"Compressing base video for RunwayML upload: {base_video_path}")
        cmd = [
            "ffmpeg", "-y", "-i", str(base_video_path),
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(compressed_base)
        ]
        subprocess.run(cmd, check=True)

    print(f"Uploading base video to RunwayML: {compressed_base}")
    try:
        upload_response = client.uploads.create_ephemeral(file=compressed_base)
        upload_uri = upload_response.uri
        print(f"Upload successful → uri: {upload_uri}")
    except Exception as e:
        print(f"Base video upload failed: {e}")
        return

    for i in range(1, num_videos):
        cam_prompt_idx = (i - 1) % len(CAMERA_PROMPTS)
        cam_prompt = CAMERA_PROMPTS[cam_prompt_idx]
        full_prompt = f"{base_prompt}. {cam_prompt}"

        cam_video_path = outputs_dir / f"cam{i:02d}.mp4"

        if cam_video_path.exists():
            print(f"⏩ RunwayML: cam{i:02d} already exists. Skipping.")
            dest_path = scene_dir / f"cam{i:02d}.mp4"
            shutil.copy2(cam_video_path, dest_path)
            print(f"✅ Copied to {dest_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[RUNWAYML Gen-4] Generating cam{i:02d} ({i}/{num_videos-1})")
        print(f"[RUNWAYML] Prompt: {full_prompt[:80]}...")
        print(f"{'='*60}\n")

        try:
            task = client.video_to_video.create(
                model="gen4_aleph",
                video_uri=upload_uri,
                prompt_text=full_prompt,
                seed=42 + i,
            )

            print(f"Task created → waiting for output...")
            result = task.wait_for_task_output()

            print("\n[DEBUG] Full task result:")
            print(result)

            video_url = None
            if hasattr(result, 'output') and isinstance(result.output, dict):
                if 'video' in result.output and isinstance(result.output['video'], dict):
                    video_url = result.output['video'].get('url')
                elif 'video_url' in result.output:
                    video_url = result.output['video_url']
                elif 'url' in result.output:
                    video_url = result.output['url']

            if video_url:
                print(f"Success! Downloading: {video_url}")
                resp = requests.get(video_url, timeout=180)
                resp.raise_for_status()
                with open(cam_video_path, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Saved {cam_video_path}")
            else:
                print("Could not find video URL in result. Check the printed output above.")

        except Exception as e:
            print(f"RunwayML generation failed for cam{i:02d}: {e}")
            continue

        if cam_video_path.exists():
            dest_path = scene_dir / f"cam{i:02d}.mp4"
            shutil.copy2(cam_video_path, dest_path)
            print(f"✅ Copied to {dest_path}")
        else:
            print(f"⚠️ cam{i:02d} was not generated")

    print(f"\n[ORCHESTRATOR] RunwayML generation complete. {num_videos} cameras processed.")


def run_vgg_preprocessor(scene_name=None, start_frame=None, end_frame=None, scale=None, fps=None, use_vggt=False, spread_frames: bool = False):
    print("Running data preprocessor...")

    # The temp version of preprocessor.py doesn't take arguments, it just processes everything in data/
    # If we want it to be specific, we might need to modify it or just run it as is.
    # However, for now let's just run it.
    # Auto-cleanup existing colmap directories to avoid conflicts
    print("Cleaning up old colmap directories...")
    colmap_pattern = os.path.join(VGG_GAUSSIAN_DIR, "data", scene_name, scene_name, "colmap_*")
    os.system(f"rm -rf {colmap_pattern}")

    cmd = ["python3", "datasets/preprocessor.py"]
    if scene_name:
        cmd.extend(["--scene", scene_name])
    
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["LD_LIBRARY_PATH"] = f"{TORCH_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PATH"] = "/root/.local/bin:" + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure site-packages and project root are in PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = f"{current_dir}:{VGG_GAUSSIAN_DIR}:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd, cwd=str(VGG_GAUSSIAN_DIR), env=env, check=True)
        print("Preprocessor finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Preprocessor error: {e}")
        raise e


def run_vgg_training(scene_name=None, duration=13):
    print("Running VGG Gaussian training...")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TORCH_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ["python3", "-u", "datasets/train_gaussian.py", "--duration", str(duration)]
    if scene_name:
        cmd.extend(["--dataset", scene_name])
    try:
        subprocess.run(cmd, cwd=str(VGG_GAUSSIAN_DIR), env=env, check=True)
        print("Validation / Training pipeline finished.")
    except subprocess.CalledProcessError as e:
        print(f"Training error: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Video Generation and 4D Gaussian Splatting.")
    parser.add_argument("--image", type=str, required=False, help="Path to base image.")
    parser.add_argument("--prompt", type=str, required=False, help="Base prompt.")
    parser.add_argument("--scene_name", type=str, default="generated_scene_01")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--video_folder", type=str, default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--use_vggt", action="store_true")
    parser.add_argument("--use_cut3r", action="store_true", help="Use CUT3R for pose estimation instead of COLMAP")
    parser.add_argument("--cut3r_model_path", type=str, default=str(WORKSPACE_ROOT / "checkpoints" / "cut3r" / "cut3r_512_dpt_4_64.pth"),
                        help="Path to CUT3R model checkpoint")
    parser.add_argument("--spread_frames", action="store_true",
                        help="Evenly sample timesteps across the full video (better parallax) instead of using consecutive frames.")
    parser.add_argument("--vace_model", type=str, default=str(DEFAULT_VACE_MODEL))
    parser.add_argument("--i2v_model", type=str, default=str(DEFAULT_I2V_MODEL))
    parser.add_argument("--cosmos_model_dir", type=str, default=str(DEFAULT_COSMOS_MODEL_DIR))
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=7.0)
    parser.add_argument("--v2v_model", type=str, choices=["vace", "cosmos", "runwayml"], default="vace",
                        help="Model for Step 2: vace (Wan2.1), cosmos, or runwayml (Gen-4 Aleph)")

    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["LD_LIBRARY_PATH"] = f"{TORCH_LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print(f"Scene Name: '{args.scene_name}'")

    if args.video_folder:
        copy_videos_from_folder(args.scene_name, args.video_folder)
    else:
        if not args.image or not args.prompt:
            print("Error: --image and --prompt required without --video_folder")
            exit(1)
            
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {args.image}")
            exit(1)

        print(f"Base Prompt: '{args.prompt}'")
        print(f"Base Image: '{img_path.absolute()}'")
        print(f"Number of Videos: {args.num_videos}")

        if args.v2v_model == "runwayml":
            print("Using RunwayML Gen-4 Aleph Video-to-Video")
            generate_runwayml_videos(
                scene_name=args.scene_name,
                base_prompt=args.prompt,
                num_videos=args.num_videos,
                sample_steps=args.sample_steps,
                guide_scale=args.guide_scale,
            )
        elif args.v2v_model == "cosmos":
            print("Using Cosmos Predict 2.5")
            generate_cosmos_videos(
                scene_name=args.scene_name,
                base_image=str(img_path.absolute()),
                base_prompt=args.prompt,
                num_videos=args.num_videos,
                cosmos_model_dir=args.cosmos_model_dir,
                sample_steps=args.sample_steps,
                guide_scale=args.guide_scale,
            )
        else:
            print("Using Wan2.1 VACE")
            generate_vace_videos(
                scene_name=args.scene_name,
                base_image=str(img_path.absolute()),
                base_prompt=args.prompt,
                num_videos=args.num_videos,
                vace_model=args.vace_model,
                i2v_model=args.i2v_model,
                sample_steps=args.sample_steps,
                guide_scale=args.guide_scale,
            )

    # Step 3: Data Preprocessing (Colmap / Pose Estimation)
    # Check if precolmap stage is already done (poses_bounds.npy exists)
    scene_data_dir = VGG_DATA_DIR / args.scene_name / args.scene_name
    poses_bounds_file = scene_data_dir / "poses_bounds.npy"
    
    use_vggt = args.use_vggt
    use_cut3r = args.use_cut3r
    
    if use_cut3r:
        # Run CUT3R pose estimation
        print(f"\n{'='*60}")
        print(f"[CUT3R] Running CUT3R pose estimation")
        print(f"{'='*60}\n")
        
        # Cleanup old colmap directories
        for colmap_dir in VGG_DATA_DIR.glob(f"{args.scene_name}/{args.scene_name}/colmap_*"):
            if colmap_dir.is_dir():
                print(f"Cleaning up old colmap directory: {colmap_dir}")
                shutil.rmtree(colmap_dir, ignore_errors=True)
        
        # Run generate_poses_bounds.py with --use_cut3r on the scene's video dir
        gen_poses_script = Path(__file__).parent / "generate_poses_bounds.py"
        cut3r_cmd = [
            "python3", str(gen_poses_script),
            "--source", str(scene_data_dir),
            "--use_cut3r",
            "--cut3r_model_path", str(args.cut3r_model_path),
        ]
        env_cut3r = os.environ.copy()
        env_cut3r["PYTHONUNBUFFERED"] = "1"
        try:
            subprocess.run(cut3r_cmd, check=True, env=env_cut3r)
            print(f"✅ CUT3R pose estimation complete")
        except subprocess.CalledProcessError as e:
            print(f"❌ CUT3R pose estimation failed: {e}")
            exit(1)
    elif use_vggt:
        # Cleanup old corrupted colmap directories to ensure fresh SfM and high point density
        for colmap_dir in VGG_DATA_DIR.glob(f"{args.scene_name}/{args.scene_name}/colmap_*"):
            if colmap_dir.is_dir():
                print(f"Cleaning up old colmap directory: {colmap_dir}")
                try:
                    shutil.rmtree(colmap_dir)
                except Exception as e:
                    print(f"Warning: Could not remove {colmap_dir}: {e}. Retrying with ignore_errors...")
                    shutil.rmtree(colmap_dir, ignore_errors=True)

        # Remove global poses_bounds.npy only if we are TRULY starting from scratch.
        # But if the user provided it, we should keep it.
        if poses_bounds_file.exists():
            print(f"✅ Found existing {poses_bounds_file}. Factoring it in (skipping removal).")

    try:
        run_vgg_preprocessor(args.scene_name, args.start_frame, args.end_frame, args.scale, args.fps, use_vggt, spread_frames=args.spread_frames)
    except Exception as e:
        print(f"Preprocessor failed: {e}")
        exit(1)

    try:
        duration = args.end_frame - args.start_frame
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{TORCH_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = f"{current_dir}:{VGG_GAUSSIAN_DIR}:{env.get('PYTHONPATH', '')}"
        cmd = ["python3", "-u", "datasets/train_gaussian.py", "--dataset", args.scene_name, "--duration", str(duration)]
        
        subprocess.run(cmd, cwd=str(VGG_GAUSSIAN_DIR), env=env, check=True)
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)

    print("Orchestration complete!")