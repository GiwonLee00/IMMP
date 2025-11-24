import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, Subset

from .THOR_dataset import THORDataset
import numpy as np
import random

class THORDatamodule():
    def __init__(self, args, cfg) -> None:
        self.args = args
        self.cfg = cfg
        
        horizon, history, plan_horizon = cfg["model"]["horizon"], cfg["model"]["history"], cfg["model"]["horizon"]
        self.train_dataset = THORDataset(data_root="data_preprocessed/THOR_processed", split="train")
        self.valid_dataset = THORDataset(data_root="data_preprocessed/THOR_processed", split="valid")

        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=cfg['dataset']['batch_size']['train'], 
                                           shuffle=cfg['dataset']['shuffle'],   
                                           num_workers=cfg['dataset']['num_workers'],
                                           pin_memory=cfg['dataset']['pin_memory'],
                                           persistent_workers=cfg['dataset']['persistent_workers'],
                                           worker_init_fn=self.seed_worker)
        self.valid_dataloader = DataLoader(self.valid_dataset, 
                                           batch_size=cfg['dataset']['batch_size']['val'], 
                                           shuffle=False,
                                           num_workers=cfg['dataset']['num_workers'],
                                           pin_memory=cfg['dataset']['pin_memory'],
                                           persistent_workers=cfg['dataset']['persistent_workers'],
                                           worker_init_fn=self.seed_worker)

    # Mixed dataset을 위한것.
    def return_train_dataset(self):
        return self.train_dataset
    
    def return_valid_dataset(self):
        return self.valid_dataset

    def return_test_dataset(self):
        return self.valid_dataset

    # single dataset을 위한것.
    def return_train_dataloader(self):
        return self.train_dataloader
    
    def return_valid_dataloader(self):
        return self.valid_dataloader

    def return_test_dataloader(self):
        return self.valid_dataloader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)