import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU(nn.Module):
    """
    ModReLU activation for complex inputs.
    
    Applies: output = ReLU(|z| + b) * (z / (|z| + eps))
    where:
        - z is the complex input
        - b is a learnable bias
        - eps is a small constant for numerical stability
    
    Args:
        num_features (int): Number of features in the input
        eps (float): Small constant to prevent division by zero
        init_bias (float): Initial value for the bias parameter
    """
    def __init__(self, num_features: int, eps: float = 1e-6, init_bias: float = 0.0):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.full((num_features,), init_bias))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ModReLU.
        
        Args:
            z (torch.Tensor): Complex input tensor
            
        Returns:
            torch.Tensor: Complex output tensor
        """
        # Calculate magnitude and phase
        mag = torch.abs(z)
        phase = z / (mag + self.eps)
        
        # Apply ReLU to magnitude with bias
        mag2 = F.relu(mag + self.bias)
        
        # Combine magnitude and phase
        return mag2 * phase