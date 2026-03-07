
import argparse
import os
from pathlib import Path
from thirdparty.gaussian_splatting.my_utils import posetow2c_matrcs, rotmat2qvec, qvec2rotmat
from script.utils_pre import write_colmap
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from pycolmap import Reconstruction
from tqdm import tqdm
import cv2
import numpy as np
import subprocess
import shutil

# from utils.colmap_utils import write_colmap
# from utils.helpers_3dg import getcolmapsinglen3d



# from utils.splats_utils import posetow2c_matrcs, qvec2rotmat, rotmat2qvec
# from thirdparty.LLFF.llff.poses.pose_utils import load_colmap_data, save_poses, gen_poses



class DatasetPreprocessor:
    
    def __init__(self, startframe=0, endframe=300, scale=1):
        self.startframe = startframe
        self.endframe = endframe
        self.scale = scale


    def extract_frames(self, video_path: Path, ext='png', save_subdir=""):
        output_path = video_path.parent / save_subdir / video_path.stem

        if all((output_path / f"{i}.{ext}").exists() for i in range(self.startframe, self.endframe)):
            print(f"Already extracted all the frames in {output_path}")
            return

        cam = cv2.VideoCapture(str(video_path))
        cam.set(cv2.CAP_PROP_POS_FRAMES, self.startframe)

        output_path.mkdir(parents=True, exist_ok=True)

        for i in range(self.startframe, self.endframe):
            ret, frame = cam.read()
            if not ret:
                break

            if self.scale > 1:
                new_width, new_height = int(frame.shape[1] / self.scale), int(frame.shape[0] / self.scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(output_path / f"{i}.{ext}"), frame)

        cam.release()


    def prepare_colmap_dnerf(self, folder, offset):
        folderlist = sorted(folder.glob("cam??/"))

        savedir = folder / f"colmap_{offset}" / "input"
        savedir.mkdir(parents=True, exist_ok=True)

        for folder in folderlist:
            imagepath = folder / f"{offset}.png"
            imagesavepath = savedir / f"{folder.name}.png"

            if imagepath.exists():
                imagesavepath.unlink(missing_ok=True)
                imagesavepath.symlink_to(imagepath.resolve())


    def get_colmapsing_len_3d(self, video_path, offset):
        getcolmapsinglen3d(video_path, offset)


    def convert_dnerf_to_colmap_db(self, video_path, offset):
        origin_numpy = video_path / "poses_bounds.npy"
        video_paths = sorted(video_path.glob("cam*.mp4"))

        with open(origin_numpy, "rb") as numpy_file:
            poses_bounds = np.load(numpy_file)

            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # Fixed: shape -> reshape

            llffposes = poses.copy().transpose(1, 2, 0)
            w2c_matriclist = posetow2c_matrcs(llffposes)
            assert (type(w2c_matriclist) == list)

            cameras = []
            for i in range(len(poses)):
                cameraname = video_paths[i].stem
                m = w2c_matriclist[i]
                colmapR = m[:3, :3]
                T = m[:3, 3]

                H, W, focal = poses[i, :, -1] / self.scale

                colmapQ = rotmat2qvec(colmapR)  # rotationMatrixToQuaternion(colmapR)

                camera = {
                    'id': i + 1,
                    'filename': f"{cameraname}.png",
                    'w': W,
                    'h': H,
                    'fx': focal,
                    'fy': focal,
                    'cx': W / 2,
                    'cy': H / 2,
                    'q': colmapQ,
                    't': T
                }
                cameras.append(camera)

            write_colmap(video_path, cameras, offset)


    def create_poses_bounds(self, video_folder):
        video_folder = Path(video_folder)
        video_files = sorted(video_folder.glob("cam*.mp4"))
        n_videos = len(video_files)
        
        # Step 1: Extract 10 frames per video directly into images/
        print("Extracting frames...")
        images_dir = video_folder / "images"
        images_dir.mkdir(exist_ok=True)
        for video in video_files:
            output_pattern = images_dir / f"{video.stem}_%04d.jpg"  # e.g., cam00_0000.jpg
            # Check if any frames exist for this video to avoid re-extraction
            if not any(images_dir.glob(f"{video.stem}_*.jpg")):
                subprocess.run([
                    "ffmpeg", "-i", str(video),
                    "-frames:v", "1",  # Extract exactly 20 frames
                    "-q:v", "2",  # High quality
                    "-loglevel", "error",
                    str(output_pattern)
                ], check=True)


        # Step 2: Run LLFF's gen_poses via imgs2poses.py
        print("Generating poses with LLFF...")
        basedir = str(video_folder)
        try:
            subprocess.run([
                "python3", "thirdparty/LLFF/imgs2poses.py", basedir
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"COLMAP failed: {e}")
            raise RuntimeError("Failed to generate poses_bounds.npy due to COLMAP error")

        # Step 3: Verify output
        poses_bounds_file = video_folder / "poses_bounds.npy"
        if not poses_bounds_file.exists():
            raise RuntimeError("poses_bounds.npy was not generated")
        poses_bounds = np.load(poses_bounds_file)
        # Note: poses_bounds.npy will have one pose per image, not per video
        print(f"Saved poses_bounds.npy with shape: {poses_bounds.shape}, dtype: {poses_bounds.dtype}")
        if poses_bounds.shape[0] < n_videos:
            print(f"Warning: Expected at least {n_videos} poses (one per video), got {poses_bounds.shape[0]}")

        # Step 4: Clean up everything except .mp4 and poses_bounds.npy
        print("Cleaning up...")
        if images_dir.exists():
            shutil.rmtree(images_dir)
        sparse_dir = video_folder / "sparse"
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        db_file = video_folder / "database.db"
        if db_file.exists():
            db_file.unlink()

    def preprocess(self, video_path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"File not found: {video_path}")
        
        # Check if poses_bounds.npy exists
        if not (video_path / "poses_bounds.npy").exists():
            self.create_poses_bounds(video_path)
        
        videolist = sorted(video_path.glob('*.mp4'))
        for video in tqdm(videolist, desc="Extract frames from videos"):
            self.extract_frames(video)

        
        for offset in range(self.startframe, self.endframe):
            self.prepare_colmap_dnerf(video_path, offset)


        for offset in tqdm(range(self.startframe, self.endframe), desc="Convert D-Nerf to Colmap"):
            self.convert_dnerf_to_colmap_db(video_path, offset)

        
        for offset in tqdm(range(self.startframe, self.endframe), desc="Running COLMAP"):
            if offset == self.startframe:
                # Clean images dir to avoid "File exists" during undistortion
                images_dir = video_path / f"colmap_{offset}" / "images"
                if images_dir.exists():
                    shutil.rmtree(images_dir)
                self.get_colmapsing_len_3d(video_path, offset)
            else:
                # Optimized Path: Reuse SfM from startframe to save time
                # We only run the undistorter for the current offset's images
                colmap_dir = video_path / f"colmap_{offset}"
                src_distorted = video_path / f"colmap_{self.startframe}" / "distorted" / "sparse"
                input_images = colmap_dir / "input"
                # Clean images dir to avoid "File exists" during undistortion
                images_dir = colmap_dir / "images"
                if images_dir.exists():
                    shutil.rmtree(images_dir)
                
                if not src_distorted.exists():
                    print(f"Warning: Source distorted model {src_distorted} not found. Running full SfM for offset {offset}.")
                    self.get_colmapsing_len_3d(video_path, offset)
                    continue

                print(f"Reusing SfM from {src_distorted} for offset {offset}")
                undist_cmd = f"colmap image_undistorter --image_path {input_images} --input_path {src_distorted} --output_path {colmap_dir} --output_type COLMAP"
                exit_code = os.system(undist_cmd)
                
                if exit_code == 0:
                    # Cleanup input like the original function
                    shutil.rmtree(input_images, ignore_errors=True)
                    # Move sparse files to sparse/0
                    sparse_root = colmap_dir / "sparse"
                    os.makedirs(sparse_root / "0", exist_ok=True)
                    for f in os.listdir(sparse_root):
                        if f != '0':
                            shutil.move(sparse_root / f, sparse_root / "0" / f)
                else:
                    print(f"Error: Undistortion failed for offset {offset}")

    
    def __call__(self, *args, **kwds):
        self.preprocess(*args, **kwds)