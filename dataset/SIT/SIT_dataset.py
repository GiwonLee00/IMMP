import os
import torch
from torch.utils.data import Dataset

class SITDataset(Dataset):
    def __init__(self, folder_paths):
        self.file_paths = []
        
        for folder in folder_paths:
            self.file_paths.extend(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
            )
        self.file_paths.sort()  # 파일들을 정렬하여 순서를 일정하게 유지.

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, weights_only=False)
        return data
