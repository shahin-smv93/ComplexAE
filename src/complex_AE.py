import torch
import torch.nn as nn
import pytorch_lightning as pl
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU, ComplexBatchNorm1d
from .extra_layers import ModReLU


class ComplexAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        bottleneck_dim: int,
        activation: str = 'modrelu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Complex-valued Autoencoder.
        
        Args:
            input_dim (int): Input dimension (number of DMD coefficients)
            hidden_dims (list): List of hidden layer dimensions
            bottleneck_dim (int): Dimension of the bottleneck layer
            activation (str): Activation function to use ('modrelu' or 'complexrelu')
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for L2 regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder layers
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(ComplexLinear(prev_dim, hidden_dim))
            encoder_layers.append(ComplexBatchNorm1d(hidden_dim))
            if activation == 'modrelu':
                encoder_layers.append(ModReLU(hidden_dim))
            else:
                encoder_layers.append(ComplexReLU())
            prev_dim = hidden_dim 
        # Bottleneck layer
        encoder_layers.append(ComplexLinear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(ComplexLinear(prev_dim, hidden_dim))
            decoder_layers.append(ComplexBatchNorm1d(hidden_dim))
            if activation == 'modrelu':
                decoder_layers.append(ModReLU(hidden_dim))
            else:
                decoder_layers.append(ComplexReLU())
            prev_dim = hidden_dim
        # Output layer
        decoder_layers.append(ComplexLinear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def shared_step(self, batch: torch.Tensor, stage: str):
        """
        Shared step for training, validation, and testing.
        
        Args:
            batch (torch.Tensor): Complex input batch
            stage (str): Current stage ('train', 'val', or 'test')
            
        Returns:
            torch.Tensor: Loss value
        """
        x = batch
        x_reconstructed = self(x)
        # Complex MSE loss
        loss = torch.mean(torch.abs(x - x_reconstructed) ** 2)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self.shared_step(batch, 'train')
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.shared_step(batch, 'val')
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.shared_step(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
