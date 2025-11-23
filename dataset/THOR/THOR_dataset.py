import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import os

class THORDataset(Dataset):
    def __init__(self, data_root=None, split=None):
        assert split in ['train', 'valid'], 'error'
        self.file_paths = f"{data_root}/{split}"
        self.len_total = len([f for f in os.listdir(self.file_paths) if os.path.isfile(os.path.join(self.file_paths, f))])

    def __len__(self):
        return self.len_total

    def __getitem__(self, idx):
        data = torch.load(f'{self.file_paths}/{idx}.pt')
        return data
        

