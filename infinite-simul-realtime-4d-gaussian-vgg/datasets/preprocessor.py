import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (infinite-simul-realtime-4d-gaussian-vgg/)
project_root = os.path.dirname(current_dir)
# Add the project root to sys.path (for third_party imports)
sys.path.insert(0, project_root)
# Add the infinite_simul_spacetime_gaussian directory to sys.path (for all submodule imports)
submodule_path = os.path.join(project_root, "third_party", "infinite_simul_spacetime_gaussian")
sys.path.insert(0, submodule_path)

# Debug prints to confirm
print(f"Current dir: {current_dir}")
print(f"Project root: {project_root}")
print(f"Full sys.path: {sys.path}")

try:
    from dataset.preprocess import DatasetPreprocessor
except ImportError:
    from third_party.infinite_simul_spacetime_gaussian.dataset.preprocess import DatasetPreprocessor
import argparse
from pathlib import Path
from config.config import start_frame, end_frame, scale

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

def process_all_datasets():
    data_dir = Path("data")
    
    # Process each dataset folder
    for dataset_folder in data_dir.iterdir():
        if dataset_folder.is_dir():
            print(f"\nProcessing dataset: {dataset_folder.name}")
            
            # Process each subfolder in the dataset
            for subfolder in dataset_folder.iterdir():
                if subfolder.is_dir():
                    process_dataset_folder(subfolder)

if __name__ == "__main__":
    process_all_datasets() 