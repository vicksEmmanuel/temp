import os
import sys
import argparse
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (infinite-simul-realtime-4d-gaussian-vgg/)
project_root = os.path.dirname(current_dir)
# Add the project root to sys.path (for third_party imports)
sys.path.insert(0, project_root)
# Add the infinite_simul_spacetime_gaussian directory to sys.path (for all submodule imports)
submodule_path = os.path.join(project_root, "third_party", "infinite_simul_spacetime_gaussian")
sys.path.insert(0, submodule_path)

try:
    from dataset.preprocess import DatasetPreprocessor
except ImportError:
    from third_party.infinite_simul_spacetime_gaussian.dataset.preprocess import DatasetPreprocessor

# config/config.py might not exist, but we expect it to
try:
    from config.config import start_frame, end_frame, scale
except ImportError:
    # Fallback defaults if config is missing
    start_frame, end_frame, scale = 0, 50, 1

def process_dataset_folder(folder_path):
    """Process a single dataset folder using the DatasetPreprocessor"""
    print(f"Processing folder: {folder_path}")

    try:
        dataset = DatasetPreprocessor(
            startframe=start_frame,
            endframe=end_frame,
            scale=scale
        )
        dataset(str(folder_path))
        print(f"Successfully processed {folder_path}")
    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

def process_all_datasets(target_scene=None):
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"Data directory {data_dir.absolute()} not found.")
        return
    
    # Process each dataset folder
    for dataset_folder in data_dir.iterdir():
        if dataset_folder.is_dir():
            if target_scene and dataset_folder.name != target_scene:
                continue
            print(f"\nProcessing dataset: {dataset_folder.name}")
            
            # Process each subfolder in the dataset
            for subfolder in dataset_folder.iterdir():
                if subfolder.is_dir():
                    process_dataset_folder(subfolder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None, help="Specific scene to process")
    args = parser.parse_args()
    process_all_datasets(target_scene=args.scene)