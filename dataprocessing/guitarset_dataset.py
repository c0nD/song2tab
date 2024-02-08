import torch
from torch.utils.data import Dataset
import numpy as np
import os


class GuitarSetDataset(Dataset):
    def __init__(self, npz_files):
        """
        Initialize the dataset with a list of npz files containing processed data.
        
        Args:
            npz_files (list of strs): A list of npz file paths.
        """
        self.npz_files = npz_files
        
        
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.npz_files)
    
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        
        Returns:
            A tuple (input_tensor, target_tensor) where:
                input_tensor (torch.Tensor): The input features for the model.
                target_tensor (torch.Tensor): The target output (e.g., tablature) for the model.
        """
        data = np.load(self.npz_files[idx])
        
        # Extract the input and target data from the npz file
        audio_features = data['audio']
        tablature_output = data['tablature']
        
        # Convert to PyTorch tensors (.long for int labels, .float for continuous)
        input_tensor = torch.from_numpy(audio_features).float()
        target_tensor = torch.from_numpy(tablature_output).long()
        
        return input_tensor, target_tensor