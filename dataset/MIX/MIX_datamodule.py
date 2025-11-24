import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import yaml
import os
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(__file__).split(os.sep)[:-2])

from dataset.crowd_nav.datamodule_crowdnav import DataModule_CrowdNav
from dataset.ETH.datamodule_eth import DataModule_ETH  
from dataset.SIT.SIT_datamodule import SITDatamodule 
from dataset.THOR.THOR_datamodule import THORDatamodule

class MIXDatamodule():
    def __init__(self, args, cfg) -> None:
        
        total_train_dataset, total_valid_dataset, total_test_dataset = [], [], []
        for merging_dataset in cfg["total_dataset_list"]:
            if merging_dataset in args.except_dataset_name:
                continue
            
            if merging_dataset == "crowd_nav":
                with open(f"configs/baseline_gametheoretic_crowd_nav.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)
            elif merging_dataset == "eth":
                with open(f"configs/baseline_gametheoretic_eth.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)    
            elif merging_dataset == "hotel":
                with open(f"configs/baseline_gametheoretic_hotel.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)    
            elif merging_dataset == "univ":
                with open(f"configs/baseline_gametheoretic_univ.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)    
            elif merging_dataset == "zara1":
                with open(f"configs/baseline_gametheoretic_zara1.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)    
            elif merging_dataset == "zara2":
                with open(f"configs/baseline_gametheoretic_zara2.yaml", "r") as f:
                    tmp_cfg = yaml.safe_load(f)           
            else:
                tmp_cfg = cfg

            cfg['dataset']['data_dir'] = tmp_cfg['dataset']['data_dir']
            cfg["dataset"]["name"] = tmp_cfg["dataset"]["name"]

            datamodule = {'crowd_nav': DataModule_CrowdNav,
                        'eth': DataModule_ETH,
                        'hotel': DataModule_ETH,
                        'univ': DataModule_ETH,
                        'zara1': DataModule_ETH,
                        'zara2': DataModule_ETH,
                        'SIT': SITDatamodule,  
                        'THOR': THORDatamodule,     
                        }[merging_dataset](args, cfg)
            
            total_train_dataset.append(datamodule.return_train_dataset())
            total_valid_dataset.append(datamodule.return_valid_dataset())
            total_test_dataset.append(datamodule.return_test_dataset())

        # dataset
        self.train_dataset = ConcatDataset(total_train_dataset)
        self.valid_dataset = ConcatDataset(total_valid_dataset)
        self.test_dataset = ConcatDataset(total_test_dataset)
        
        print(f"Train Dataset Size: {len(self.train_dataset)}")
        print(f"Valid Dataset Size: {len(self.valid_dataset)}")
        print(f"Test Dataset Size: {len(self.test_dataset)}")
        # breakpoint()
        # for dataset in total_train_dataset:
        #     print(f"Dataset length: {len(dataset)}")
        #     print(f"Sample shape: {dataset[0]}")

        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=cfg['dataset']['batch_size']['train'], 
                                           shuffle=cfg['dataset']['shuffle'],
                                           num_workers=cfg['dataset']['num_workers'],
                                           pin_memory=cfg['dataset']['pin_memory'],
                                           persistent_workers=cfg['dataset']['persistent_workers'],
                                           worker_init_fn=self.seed_worker)
                                           # collate_fn=self.custom_collate_fn)
        self.valid_dataloader = DataLoader(self.valid_dataset, 
                                           batch_size=cfg['dataset']['batch_size']['val'], 
                                           shuffle=False,
                                           num_workers=cfg['dataset']['num_workers'],
                                           pin_memory=cfg['dataset']['pin_memory'],
                                           persistent_workers=cfg['dataset']['persistent_workers'],
                                           worker_init_fn=self.seed_worker)
                                           # collate_fn=self.custom_collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, 
                                           batch_size=cfg['dataset']['batch_size']['test'], 
                                           shuffle=False,
                                           num_workers=cfg['dataset']['num_workers'],
                                           pin_memory=cfg['dataset']['pin_memory'],
                                           persistent_workers=cfg['dataset']['persistent_workers'],
                                           worker_init_fn=self.seed_worker)
                                           # collate_fn=self.custom_collate_fn)

    # Mixed dataset을 위한것.
    def return_train_dataset(self):
        return self.train_dataset
    
    def return_valid_dataset(self):
        return self.valid_dataset

    def return_test_dataset(self):
        return self.test_dataset

    # single dataset을 위한것.
    def return_train_dataloader(self):
        return self.train_dataloader

    def return_valid_dataloader(self):
        return self.valid_dataloader

    def return_test_dataloader(self):
        return self.test_dataloader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)