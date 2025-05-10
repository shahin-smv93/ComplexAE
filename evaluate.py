import os
import torch
import numpy as np
from pathlib import Path
import yaml
from src.train import load_config
from src.complex_AE import ComplexAutoencoder
from src.data_module import ComplexAutoencoderDataModule


def evaluate_model(
    model_path: str,
    config_path: str,
    data_path: str,
    output_dir: str = 'evaluation_results'
):
    """
    Evaluate a trained complex autoencoder model using the validation dataloader.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        config_path (str): Path to the config file
        data_path (str): Path to the data file
        output_dir (str): Directory to save evaluation results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Load data
    print(f"Loading data from: {data_path}")
    dmd_coefficients = np.load(data_path)
    print(f"Data shape: {dmd_coefficients.shape}")
    
    # Initialize data module
    data_module = ComplexAutoencoderDataModule(
        data=dmd_coefficients,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        have_test=config['data']['have_test'],
        batch_size=config['training']['batch_size'],
        shuffle=False,  # No shuffling for evaluation
        random_state=config['random_seed']
    )
    
    # Initialize model
    model = ComplexAutoencoder(
        input_dim=dmd_coefficients.shape[1],
        hidden_dims=config['model']['hidden_dims'],
        bottleneck_dim=config['model']['bottleneck_dim'],
        activation=config['model']['activation'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Load trained weights
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Setup data module and get validation dataloader
    data_module.setup('fit')
    val_dataloader = data_module.val_dataloader()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    model.cuda()  # Move model to GPU if available
    
    all_reconstructions = []
    all_originals = []
    total_loss = 0
    total_real_loss = 0
    total_imag_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to GPU
            batch = batch.cuda()
            
            # Get reconstructions
            reconstructions = model(batch)
            
            # Calculate losses
            real_diff = torch.real(batch - reconstructions)
            imag_diff = torch.imag(batch - reconstructions)
            real_loss = torch.mean(real_diff ** 2)
            imag_loss = torch.mean(imag_diff ** 2)
            batch_loss = real_loss + imag_loss
            
            # Accumulate losses
            total_loss += batch_loss.item()
            total_real_loss += real_loss.item()
            total_imag_loss += imag_loss.item()
            num_batches += 1
            
            # Store all reconstructions and originals
            all_reconstructions.append(reconstructions.cpu().numpy())
            all_originals.append(batch.cpu().numpy())
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_real_loss = total_real_loss / num_batches
    avg_imag_loss = total_imag_loss / num_batches
    
    print(f"\nValidation Results:")
    print(f"Average Total Loss: {avg_loss:.6f}")
    print(f"Average Real Part Loss: {avg_real_loss:.6f}")
    print(f"Average Imaginary Part Loss: {avg_imag_loss:.6f}")
    
    # Concatenate all reconstructions and originals
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
    print(f"\nTotal number of validation samples: {len(all_originals)}")
    print(f"Original data shape: {all_originals.shape}")
    print(f"Reconstructed data shape: {all_reconstructions.shape}")
    
    # Save all reconstructions
    np.save(os.path.join(output_dir, 'all_reconstructions.npy'), all_reconstructions)
    np.save(os.path.join(output_dir, 'all_originals.npy'), all_originals)
    
    # Save example reconstructions (first 5) for visualization
    examples = {
        'original': all_originals[:5],
        'reconstructed': all_reconstructions[:5]
    }
    np.save(os.path.join(output_dir, 'reconstruction_examples.npy'), examples)
    
    # Save the normalizer for later use
    normalizer = data_module.get_normalizer()
    np.save(os.path.join(output_dir, 'normalizer_mean.npy'), normalizer.mean)
    np.save(os.path.join(output_dir, 'normalizer_std.npy'), normalizer.std)
    
    print(f"\nSaved all reconstructions to: {output_dir}/all_reconstructions.npy")
    print(f"Saved all originals to: {output_dir}/all_originals.npy")
    print(f"Saved example reconstructions to: {output_dir}/reconstruction_examples.npy")
    print(f"Saved normalizer parameters to: {output_dir}/normalizer_*.npy")


if __name__ == '__main__':
    # Example usage
    model_path = 'checkpoints/complex_ae-last.ckpt'  # Path to your saved model
    config_path = 'config.yaml'
    data_path = 'data/data.npy'
    output_dir = 'evaluation_results'
    
    evaluate_model(
        model_path=model_path,
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir
    ) 