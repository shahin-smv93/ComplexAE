import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
import os

from .data_module import ComplexAutoencoderDataModule
from .complex_AE import ComplexAutoencoder


def load_config(config_path: str):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_complex_autoencoder(
    dmd_coefficients: np.ndarray,
    config: Dict[str, Any],
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    experiment_name: Optional[str] = None
) -> tuple[ComplexAutoencoder, ComplexAutoencoderDataModule]:
    """
    Train a complex-valued autoencoder on DMD coefficients.
    
    Args:
        dmd_coefficients (np.ndarray): Complex-valued DMD coefficients
        config (Dict[str, Any]): Training configuration
        checkpoint_dir (str): Directory to save model checkpoints
        log_dir (str): Directory to save training logs
        experiment_name (Optional[str]): Name of the experiment for logging
        
    Returns:
        tuple[ComplexAutoencoder, ComplexAutoencoderDataModule]: Trained model and data module
    """
    pl.seed_everything(int(config['random_seed']))
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    data_module = ComplexAutoencoderDataModule(
        data=dmd_coefficients,
        train_split=float(config['data']['train_split']),
        val_split=float(config['data']['val_split']),
        have_test=bool(config['data']['have_test']),
        batch_size=int(config['training']['batch_size']),
        shuffle=bool(config['data']['shuffle']),
        random_state=int(config['random_seed'])
    )
    
    model = ComplexAutoencoder(
        input_dim=dmd_coefficients.shape[1],
        hidden_dims=[int(dim) for dim in config['model']['hidden_dims']],
        bottleneck_dim=int(config['model']['bottleneck_dim']),
        activation=str(config['model']['activation']),
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='complex_ae-{epoch:02d}-{val_loss:.6f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=int(config['training']['early_stopping_patience']),
            min_delta=float(config['training']['early_stopping_min_delta']),
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name or 'complex_autoencoder',
        version=None
    )
    
    trainer = pl.Trainer(
        max_epochs=int(config['training']['max_epochs']),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=int(config['training']['log_every_n_steps']),
        deterministic=True,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, data_module)
    
    if config['data']['have_test']:
        trainer.test(model, data_module)
    
    return model, data_module


def main():
    """
    Example usage of the complex autoencoder training.
    This is just a template showing how to use the training function.
    In practice, you would:
    1. Load your configuration from config.yaml
    2. Load your DMD coefficients
    3. Call train_complex_autoencoder() with your data and config
    """
    # Load configuration from yaml file
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Load your DMD coefficients
    data_path = os.path.join(config['paths']['data_dir'], config['paths']['data_file'])
    dmd_coefficients = np.load(data_path)
    
    # Train the model
    model, data_module = train_complex_autoencoder(
        dmd_coefficients=dmd_coefficients,
        config=config,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir'],
        experiment_name='dmd_autoencoder'
    )


if __name__ == '__main__':
    main()
