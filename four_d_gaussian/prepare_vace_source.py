#!/usr/bin/env python3
"""
Prepare a source video and mask for VACE first-frame conditioning.

Given an input image, creates:
  1. A source video (mp4) where frame 0 = the input image, rest = black
  2. A mask video (mp4) where frame 0 = black (keep), rest = white (generate)

This tells VACE: "keep the first frame as-is and generate the remaining frames."
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def prepare_vace_source(image_path: str, output_dir: str, num_frames: int = 81,
                        width: int = 832, height: int = 480, fps: int = 16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and resize the input image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    img_array = np.array(img)  # RGB
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Create source video: frame 0 = image, rest = black
    src_video_path = str(output_dir / "src_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(src_video_path, fourcc, fps, (width, height))
    writer.write(img_bgr)  # Frame 0: the input image
    black_frame = np.zeros_like(img_bgr)
    for _ in range(num_frames - 1):
        writer.write(black_frame)  # Frames 1-N: black
    writer.release()

    # Create mask video: frame 0 = black (keep), rest = white (generate)
    mask_video_path = str(output_dir / "src_mask.mp4")
    writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))
    writer.write(black_frame)  # Frame 0: black = "keep this frame"
    white_frame = np.ones_like(img_bgr) * 255
    for _ in range(num_frames - 1):
        writer.write(white_frame)  # Frames 1-N: white = "generate these"
    writer.release()

    print(f"✅ Source video: {src_video_path}")
    print(f"✅ Mask video:   {mask_video_path}")
    return src_video_path, mask_video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    prepare_vace_source(args.image, args.output_dir, args.num_frames,
                        args.width, args.height, args.fps)
