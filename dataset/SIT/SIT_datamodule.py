import torch
from torch.utils.data import Dataset, DataLoader
import os

from .SIT_dataset import SITDataset
import random
import numpy as np

def get_folders_with_keywords(root_path, keywords):
    matching_folders = []
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            # Check if any keyword is in the folder name
            if any(keyword in dirname for keyword in keywords):
                matching_folders.append(os.path.join(dirpath, dirname))
    return matching_folders

class SITDatamodule():
    def __init__(self, args, cfg) -> None:
        # NOTE: base path에 second processed folder가 위치 
        self.using_human_as_robot = False
        base_path = "data_preprocessed/SIT_processed_3rd"
        
        train_folder_keywords = [
            "Cafe_street_1", "Cafeteria_6", "Cafeteria_2", "Cafeteria_5", "Cafeteria_1", "Corridor_1", "Corridor_8", "Corridor_9", "Corridor_10", "Corridor_5", "Corridor_11", "Corridor_3", "Courtyard_4", "Courtyard_9", "Courtyard_8",
            "Courtyard_5", "Courtyard_6", "Crossroad_1", "Hallway_1", "Hallway_8", "Hallway_6", "Hallway_11", "Hallway_7", "Hallway_4", "Hallway_2", "Lobby_2", "Lobby_5", "Lobby_3", "Lobby_8", "Lobby_6",
            "Outdoor_Alley_2", "Subway_Entrance_2", "Subway_Entrance_4", "Three_way_Intersection_3", "Three_way_Intersection_8"
        ]

        valid_folder_keywords = [
            "Cafe_street_2", "Cafeteria_3", "Corridor_7", "Corridor_2", "Courtyard_2", "Courtyard_1", "Hallway_10", "Hallway_9",
            "Hallway_3", "Lobby_7", "Lobby_4", "Outdoor_Alley_3", "Three_way_Intersection_5", "Three_way_Intersection_4"
        ]

        test_folder_keywords = [
            "Cafe_street_2", "Cafeteria_3", "Corridor_7", "Corridor_2", "Courtyard_2", "Courtyard_1", "Hallway_10", "Hallway_9",
            "Hallway_3", "Lobby_7", "Lobby_4", "Outdoor_Alley_3", "Three_way_Intersection_5", "Three_way_Intersection_4"
        ]

        if args.exp_with_specific_scene == "Yes":
            print(f"Only using {args.exp_scene_name} scene!!")
            breakpoint()
            train_folder_keywords = [folder for folder in train_folder_keywords if args.exp_scene_name in folder]
            valid_folder_keywords = [folder for folder in valid_folder_keywords if args.exp_scene_name in folder]
            test_folder_keywords = [folder for folder in test_folder_keywords if args.exp_scene_name in folder]

        '''
        valid_folder_keywords = [
            "Cafe_street_2", "Cafeteria_3", "Corridor_7", "Corridor_2", "Courtyard_2", "Courtyard_1", "Hallway_10", "Hallway_9",
            "Hallway_3", "Lobby_7", "Lobby_4", "Outdoor_Alley_3", "Three_way_Intersection_5", "Three_way_Intersection_4"
        ]
        '''
        '''
        test_folder_keywords =[
            "Cafeteria_4", "Corridor_6", "Corridor_4", "Courtyard_3", "Courtyard_7", "Hallway_5", "Hallway_12", "Lobby_1", "Lobby_9", "Outdoor_Alley_1",
            "Subway_Entrance_1", "Subway_Entrance_3", "Three_way_Intersection_6", "Three_way_Intersection_7"
        ]
        '''
        train_folders, valid_folders, test_folders = [], [], []

        for current_folder, current_keywords in zip([train_folders, valid_folders, test_folders], [train_folder_keywords, valid_folder_keywords, test_folder_keywords]):
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path) and any(keyword in folder for keyword in current_keywords):
                    current_folder.append(folder_path)

        self.train_folders = train_folders
        self.valid_folders = valid_folders
        self.test_folders = test_folders

        # dataset
        self.train_dataset = SITDataset(folder_paths=self.train_folders)
        self.valid_dataset = SITDataset(folder_paths=self.valid_folders)
        self.test_dataset = SITDataset(folder_paths=self.test_folders)
        
        print(f"Train Dataset Size: {len(self.train_dataset)}")
        print(f"Valid Dataset Size: {len(self.valid_dataset)}")
        print(f"Test Dataset Size: {len(self.test_dataset)}")

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

    