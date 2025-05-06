import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

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
    pl.seed_everything(config['random_seed'])
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    data_module = ComplexAutoencoderDataModule(
        data=dmd_coefficients,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        have_test=config['data']['have_test'],
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle'],
        random_state=config['random_seed']
    )
    
    model = ComplexAutoencoder(
        input_dim=dmd_coefficients.shape[1],
        hidden_dims=config['model']['hidden_dims'],
        bottleneck_dim=config['model']['bottleneck_dim'],
        activation=config['model']['activation'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='complex_ae-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name or 'complex_autoencoder',
        version=None
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['training']['log_every_n_steps'],
        deterministic=True
    )
    
    trainer.fit(model, data_module)
    
    if config['data']['have_test']:
        trainer.test(model, data_module)
    
    return model, data_module


def main():
    # Example configuration
    config = {
        'random_seed': 42,
        'data': {
            'train_split': 0.7,
            'val_split': 0.2,
            'have_test': True,
            'shuffle': True
        },
        'model': {
            'hidden_dims': [64, 32],
            'bottleneck_dim': 16,
            'activation': 'modrelu'  # or 'complexrelu'
        },
        'training': {
            'batch_size': 32,
            'max_epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,
            'log_every_n_steps': 10
        }
    }
    
    # Example usage
    # Load your DMD coefficients here
    # dmd_coefficients = np.load('path_to_your_dmd_coefficients.npy')
    
    # Train the model
    # model, data_module = train_complex_autoencoder(
    #     dmd_coefficients=dmd_coefficients,
    #     config=config,
    #     checkpoint_dir='checkpoints',
    #     log_dir='logs',
    #     experiment_name='dmd_autoencoder'
    # )


if __name__ == '__main__':
    main()
