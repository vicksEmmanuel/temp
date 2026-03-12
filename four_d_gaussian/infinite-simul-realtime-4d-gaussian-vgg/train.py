from datasets.loader import VideoPLYDataModule

# Load the dataset
data_module = VideoPLYDataModule(data_root='data', batch_size=4, num_workers=0)
data_module.setup()

# Get the first batch
videos, ply, gaussian_models = next(iter(data_module.train_dataloader()))
print(f'Videos: {videos}')
print(f'Plies {ply}')
print(f"Gaussian {gaussian_models}")