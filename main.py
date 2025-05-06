import os
import numpy as np
from pathlib import Path
import yaml
import torch

torch.set_float32_matmul_precision('medium')

from src.train import train_complex_autoencoder, load_config


def main():
    config = load_config('config.yaml')
    
    for dir_path in [
        config['paths']['checkpoint_dir'],
        config['paths']['log_dir'],
        config['paths']['data_dir']
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    data_path = os.path.join(config['paths']['data_dir'], config['paths']['data_file'])
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    dmd_coefficients = np.load(data_path)
    
    print(f"Loaded DMD coefficients shape: {dmd_coefficients.shape}")
    
    model, data_module = train_complex_autoencoder(
        dmd_coefficients=dmd_coefficients,
        config=config,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir'],
        experiment_name='dmd_autoencoder'
    )
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main() 