import subprocess
from pathlib import Path
import os
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Define the infinite_simul_spacetime_gaussian directory
train_dir = os.path.join(project_root, "third_party", "infinite_simul_spacetime_gaussian")

def train_colmap_folders(folder_path, duration=50):
    """Find and process only colmap_0 folders in the given path for 4D training"""
    base_path = Path(folder_path)
    
    # Find colmap_0 folders (we only start training from frame 0 in 4D)
    colmap_folders = []
    for path in base_path.rglob("*"):
        if path.is_dir() and path.name == 'colmap_0':
            colmap_folders.append(path)
    
    if not colmap_folders:
        print(f"No colmap_0 folder found in {folder_path}. Searching for any colmap_ folder as fallback...")
        for path in base_path.rglob("*"):
            if path.is_dir() and path.name.startswith('colmap_'):
                colmap_folders.append(path)
                break # Just take the first one
    
    for colmap_folder in colmap_folders:
        print(f"Initiating 4D training from {colmap_folder} with duration {duration}")
        
        # Construct absolute paths
        source_path = colmap_folder.resolve()  # Absolute path to colmap_ folder
        model_path = (source_path / "log").resolve()  # Absolute path to log folder
        config_path = os.path.join(train_dir, "configs", "general.json")
        
        print(f"Source Path: {source_path}")
        print(f"Model Path: {model_path}")
        print(f"Config Path: {config_path}")
        
        def training_proceed():
            print(f"{model_path} does not exist or needs restart, proceeding with training")
            # Construct the shell command to cd and run train.py
            
            cmd = (
                f"cd {train_dir} && python3 train.py "
                f"--model_path {model_path} "
                f"--source_path {source_path} "
                f"--config {config_path} "
                f"--duration {duration}"
            )
            
            try:
                # Run the command in a shell
                subprocess.run(cmd, check=True, shell=True)
                print(f"Successfully finished training for {colmap_folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {colmap_folder}: {e}")
        
        # For 4D sessions, we usually want to overwrite or restart if incomplete
        if model_path.exists():
            ply_path = model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            if ply_path.exists():
                print(f"Skipping {colmap_folder}: {ply_path} already exists")
                continue
            else:
                print(f"Restarting training in {model_path}")
                # We don't necessarily need to delete if it supports resuming, 
                # but to be safe we'll clean and start fresh if it's corrupted.
                shutil.rmtree(model_path, ignore_errors=True)
        
        training_proceed()

def process_all_datasets(dataset=None, duration=50):
    data_dir = Path("data")
    
    if dataset:
        dataset_folder = data_dir / dataset
        if dataset_folder.is_dir():
            print(f"\nProcessing dataset: {dataset_folder.name}")
            train_colmap_folders(dataset_folder, duration=duration)
        else:
            print(f"Dataset folder {dataset_folder} not found.")
    else:
        # Process each dataset folder
        for dataset_folder in data_dir.iterdir():
            if dataset_folder.is_dir():
                print(f"\nProcessing dataset: {dataset_folder.name}")
                train_colmap_folders(dataset_folder, duration=duration)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--duration", type=int, default=50)
    args = parser.parse_args()
    
    process_all_datasets(dataset=args.dataset, duration=args.duration)