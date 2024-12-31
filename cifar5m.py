import numpy as np
import torch
import glob
from torch.utils.data import Dataset

# CIFAR-5M Dataset Loader
class CIFAR5MDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, seed=0):
        self.data_files = sorted(glob.glob(f"{data_dir}/part*.npz"))
        self.transform = transform
        self.seed = seed
        self.data, self.labels = self.load_data(train)
    
    def load_data(self, train):
        data, labels = [], []
        for file in self.data_files:
            npz_data = np.load(file)
            data.append(npz_data['X'] / 255.0)
            labels.append(npz_data['Y'])
        
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.random.seed(self.seed)
        idx = np.random.permutation(len(data))
        data, labels = data[idx], labels[idx]

        split_point = 5_000_000
        return (data[:split_point], labels[:split_point]) if train else (data[split_point:], labels[split_point:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert numpy array to PyTorch tensor
        image = torch.tensor(self.data[idx], dtype=torch.float32).permute(2, 0, 1)  # Shape to (C, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
        
        return image, label