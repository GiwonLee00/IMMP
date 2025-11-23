import torch
from torch.utils.data import DataLoader
from dotdict import dotdict

from .dataset_eth_v3 import ForecastDataset_ETH
import numpy as np
import random

class DataModule_ETH():
    def __init__(self, args, cfg) -> None:
        self.args = args
        self.cfg = cfg

        horizon, history, plan_horizon = cfg["model"]["horizon"], cfg["model"]["history"], cfg["model"]["horizon"]
        # dataset = ForecastDataset_ETH(cfg['dataset'])
        # dataset = ForecastDataset_ETH(cfg['dataset']['data_dir'], horizon=horizon, history=history, plan_horizon=plan_horizon)
        
        train_parser = {'split': 'train', 'past_frames': history, 'future_frames': horizon, 'min_past_frames': history, \
            'frame_skip': 1, 'data_root': '../data/eth_ucy', 'dataset': cfg["dataset"]["name"], 'min_future_frames': horizon, 'traj_scale': 1, \
                'load_map': False}
        train_parser = dotdict(train_parser)
        # breakpoint()
        self.train_dataset = ForecastDataset_ETH(train_parser, split='train', data_root="data_preprocessed/ETH_UCY_processed")
        self.valid_dataset = ForecastDataset_ETH(train_parser, split='val', data_root="data_preprocessed/ETH_UCY_processed")

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