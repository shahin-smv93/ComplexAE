import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ComplexNormalizer:
    def __init__(self):
        self.std = None
        self.eps = 1e-6
        
    def fit(self, data: np.ndarray):
        """
        Fit the normalizer to the data.
        
        Args:
            data (np.ndarray): Complex-valued data array of shape (n_samples, n_features)
        """
        # Calculate std of magnitudes with numerical stability
        magnitudes = np.abs(data)  # shape (N, D)
        # Add small epsilon to prevent zero magnitudes
        magnitudes = magnitudes + self.eps
        self.std = np.std(magnitudes, axis=0)
        # Ensure std is at least eps
        self.std = np.maximum(self.std, self.eps)
        
    def transform(self, data: np.ndarray):
        """
        Normalize by dividing each complex feature by its magnitude std.
        
        Args:
            data (np.ndarray): Complex-valued data array to normalize
            
        Returns:
            np.ndarray: Normalized complex-valued data
        """
        if self.std is None:
            raise ValueError("Call fit() before transform()")
        # Add small epsilon to prevent division by zero
        return data / (self.std + self.eps)
            
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Recover original scale by multiplying by std.
        
        Args:
            data (np.ndarray): Normalized complex-valued data
            
        Returns:
            np.ndarray: Original scale complex-valued data
        """
        if self.std is None:
            raise ValueError("Call fit() before inverse_transform()")
        return data * self.std


class DatasetGen:
    """
    A class for generating datasets from a numpy array.
    Args:
        data (np.ndarray): Complex-valued data array of shape (n_samples, n_features)
        train_split (float): Proportion of data to use for training
        val_split (float): Proportion of data to use for validation
        have_test (bool): Whether to include a test set
        shuffle (bool): Whether to shuffle the data
        random_state (int): Random seed for reproducibility
    """

    def __init__(self, data, train_split, val_split, have_test: bool = False, shuffle: bool = False, random_state: int = 42):
        self.data = data
        self.train_split = train_split
        self.val_split = val_split
        self.have_test = have_test
        self.shuffle = shuffle
        self.random_state = random_state

        if not have_test:
            assert abs(train_split + val_split - 1) < 1e-10
            self.test_split = 0
        else:
            self.test_split = 1 - train_split - val_split
            assert self.test_split > 0
        
        assert abs(self.train_split + self.val_split + self.test_split - 1) < 1e-10

        self.split_data()

    def split_data(self):
        """
        Split the data into train, val, and test sets.
        """
        dataset = self.data
        
        if self.have_test:
            train_len = int(self.train_split * len(dataset))
            temp_len = len(dataset) - train_len
            val_len = int(self.val_split * temp_len)
            test_len = temp_len - val_len
            assert abs(train_len + val_len + test_len - len(dataset)) < 1e-10

            train_dataset, temp_dataset = train_test_split(dataset, train_size=train_len, shuffle=self.shuffle, random_state=self.random_state)
            val_dataset, test_dataset = train_test_split(temp_dataset, train_size=val_len, shuffle=self.shuffle, random_state=self.random_state)
            self.test_dataset = test_dataset
        else:
            train_len = int(self.train_split * len(dataset))
            train_dataset, val_dataset = train_test_split(dataset, train_size=train_len, shuffle=self.shuffle, random_state=self.random_state)
            self.test_dataset = None
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_slice(self, split: str):
        """
        Get a slice of the data.
        Args:
            split (str): One of 'train', 'val', or 'test'
        Returns:
            np.ndarray: Data slice
        """
        assert split in ['train', 'val', 'test']
        if split == 'train':
            return self.train_dataset
        elif split == 'val':
            return self.val_dataset
        else:
            return self.test_dataset

    @property
    def train_data(self):
        return self.train_dataset
    
    @property
    def val_data(self):
        return self.val_dataset
    
    @property
    def test_data(self):
        return self.test_dataset

    def length(self, split):
        data = {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset),
            'test': len(self.test_dataset) if self.test_dataset is not None else 0
        }
        return data[split]


class ComplexDataset(Dataset):
    def __init__(self, dataset_gen: DatasetGen, split: str, normalizer: ComplexNormalizer = None, split_complex: bool = False):
        """
        A PyTorch Dataset for complex-valued data.
        
        Args:
            dataset_gen (DatasetGen): The dataset generator containing the split data
            split (str): One of 'train', 'val', or 'test'
            normalizer (ComplexNormalizer, optional): Normalizer for complex data
            split_complex (bool): If True, splits complex numbers into real/imaginary parts.
                                If False, keeps them as complex numbers (default for complexPyTorch).
        """
        self.data = dataset_gen.get_slice(split)
        self.normalizer = normalizer
        self.split_complex = split_complex
        
        if self.normalizer is not None:
            self.data = self.normalizer.transform(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        complex_data = self.data[idx]
        
        if self.split_complex:
            # split into real and imaginary parts
            real_part = np.real(complex_data)
            imag_part = np.imag(complex_data)
            
            # stack real and imaginary parts along first dimension to achieve shape: (2, n_features)
            return torch.tensor(np.stack([real_part, imag_part], axis=0), dtype=torch.float32)
        else:
            # keep as complex numbers to achieve shape: (n_features,)
            return torch.tensor(complex_data, dtype=torch.complex64)


class ComplexAutoencoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: np.ndarray,
        train_split: float = 0.8,
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        random_state: int = 42,
        have_test: bool = False
    ):
        """
        DataModule for complex autoencoder.
        
        Args:
            data (np.ndarray): Complex-valued data array of shape (n_samples, n_features)
            train_split (float): Proportion of data to use for training
            val_split (float): Proportion of data to use for validation
            batch_size (int): Batch size for DataLoader
            num_workers (int): Number of workers for DataLoader
            shuffle (bool): Whether to shuffle the data
            random_state (int): Random seed for reproducibility
            have_test (bool): Whether to include a test set
        """
        super().__init__()
        self.data = data
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.random_state = random_state
        self.have_test = have_test
        
        # initialize normalizer
        self.normalizer = ComplexNormalizer()
        
        # initialize dataset generator
        self.dataset_gen = DatasetGen(
            data=self.data,
            train_split=self.train_split,
            val_split=self.val_split,
            have_test=self.have_test,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
    def setup(self, stage: str = None):
        """Setup datasets for each split"""
        if stage == 'fit' or stage is None:
            # fit normalizer on training data
            train_data = self.dataset_gen.get_slice('train')
            self.normalizer.fit(train_data)
            
            # create datasets with normalization
            self.train_dataset = ComplexDataset(self.dataset_gen, 'train', self.normalizer)
            self.val_dataset = ComplexDataset(self.dataset_gen, 'val', self.normalizer)
            
        if stage == 'test' or stage is None:
            if self.have_test:
                # create test dataset with normalization
                self.test_dataset = ComplexDataset(self.dataset_gen, 'test', self.normalizer)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def test_dataloader(self):
        if not self.have_test:
            raise ValueError("Test dataset not available. Set have_test=True during initialization.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def get_normalizer(self):
        """Get the fitted normalizer for use in the model"""
        return self.normalizer