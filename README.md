# Complex-Valued Autoencoder for DMD Coefficients

This project implements a complex-valued autoencoder for Dynamic Mode Decomposition (DMD) coefficients using PyTorch Lightning and complexPyTorch.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `config.yaml`: Configuration file for training
- `main.py`: Main script to run training
- `src/`: Source code directory
  - `data_module.py`: Data handling
  - `complex_AE.py`: Autoencoder model
  - `extra_layers.py`: Custom layers
  - `train.py`: Training module
- `data/`: Directory for input data
- `checkpoints/`: Directory for saved models
- `logs/`: Directory for training logs

## Usage

1. Place your DMD coefficients in the `data/` directory as `dmd_coefficients.npy`

2. Configure training parameters in `config.yaml`:
   - Model architecture
   - Training parameters
   - Data splits
   - Directory paths

3. Run training:
```bash
python main.py
```

4. Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

## Features

- Complex-valued autoencoder using complexPyTorch
- ModReLU activation for complex numbers
- Proper handling of complex-valued data
- Training with PyTorch Lightning
- Model checkpointing and early stopping
- TensorBoard logging
- Configuration via YAML file 