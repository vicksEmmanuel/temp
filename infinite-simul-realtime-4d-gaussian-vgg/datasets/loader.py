# datasets/loader.py
from argparse import ArgumentParser
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Add all necessary paths
base_path = os.path.dirname(os.path.dirname(__file__))
gaussian_path = os.path.join(base_path, "third_party/infinite_simul_spacetime_gaussian/thirdparty/gaussian_splatting")
sys.path.append(os.path.join(base_path, "third_party/infinite_simul_spacetime_gaussian"))
sys.path.append(os.path.join(base_path, "third_party/infinite_simul_spacetime_gaussian/thirdparty"))
sys.path.append(gaussian_path)

# Now import with the correct paths
from thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
from thirdparty.gaussian_splatting.arguments import ModelParams


class VideoPLYDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} valid samples across all datasets")
        
    def _collect_samples(self) -> List[Tuple[List[Path], Path]]:
        samples = []
        
        # Iterate through xxx folders (e.g., Neural_3D_Dataset)
        for xxx_folder in self.data_root.iterdir():
            if not xxx_folder.is_dir():
                continue
                
            # Iterate through yyy folders (e.g., coffee_martini, fire_martini)
            for yyy_folder in xxx_folder.iterdir():
                if not yyy_folder.is_dir():
                    continue
                    
                # Get video files specific to this yyy_folder
                video_files = [f for f in yyy_folder.iterdir() 
                             if f.is_file() and f.suffix in ['.mp4', '.avi', '.mov']]
                
                print(f"Found {len(video_files)} video files in {yyy_folder}")
                
                if not video_files:
                    print(f"Skipping {yyy_folder}: no video files")
                    continue
                    
                # Get colmap folders specific to this yyy_folder
                colmap_folders = [f for f in yyy_folder.iterdir() 
                                if f.is_dir() and f.name.startswith('colmap_')]
                
                # Filter to colmap folders with the specific PLY file at log/point_cloud/iteration_30000/point_cloud.ply
                valid_colmap_folders = []
                for colmap_folder in colmap_folders:
                    ply_path = colmap_folder / "log" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                    if ply_path.exists():
                        valid_colmap_folders.append((colmap_folder, ply_path))
                    else:
                        print(f"No PLY file found at {ply_path}")
                
                print(f"Found {len(valid_colmap_folders)} valid colmap folders with PLY files in {yyy_folder}")
                
                if not valid_colmap_folders:
                    print(f"Skipping {yyy_folder}: no valid colmap folders with PLY files")
                    continue
                    
                # Create samples only within this yyy_folder
                for num_videos in range(1, min(6, len(video_files) + 1)):
                    for _ in range(3):
                        selected_videos = random.sample(video_files, num_videos)
                        colmap_folder, ply_path = random.choice(valid_colmap_folders)
                        samples.append((selected_videos, ply_path))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[List[Path], Path]:
        videos, ply = self.samples[idx]
        sh_degree = ModelParams(parser=ArgumentParser(
                description="Training script parameters"
            )
        ).sh_degree
        
        gaussian_model = GaussianModel(sh_degree)
        print(f"Ply to load {ply}")
        gaussian_model.load_ply(Path(ply).__str__())
        
        return videos, ply, gaussian_model

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length video lists."""
    videos = [item[0] for item in batch]  # List of video lists
    plies = [item[1] for item in batch]   # List of PLY paths
    gaussian_model = [item[2] for item in batch]  # List of GaussianModel objects
    return videos, plies, gaussian_model

class VideoPLYDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = VideoPLYDataset(self.data_root)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn
        )