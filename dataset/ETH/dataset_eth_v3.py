'''
v2: 
- 모든 agent local coordinates, agent-agent / agent-robt 거리를 추가적인 벡터로 사용
- input position, output difference
'''
import tqdm
import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.append("/mnt/minseok/APTP")
from utils.eth_utils import preprocess
from utils.eth_utils import get_ethucy_split
import random
import copy
import os

class ForecastDataset_ETH(Dataset):
    def __init__(self, parser, split='train', frame_skip=1, data_root=None):
        self.split = split
        assert split in ['train', 'val', 'test'], 'error'
        self.file_paths = f"{data_root}/{parser.dataset}/{split}"
        self.len_total = len([f for f in os.listdir(self.file_paths) if os.path.isfile(os.path.join(self.file_paths, f))])

    def __len__(self):
        return self.len_total

    def __getitem__(self, idx):
        data = torch.load(f'{self.file_paths}/{idx}.pt')
        return data

    def visualization_function_dataset(self, neg_seeds, neg_hist, neg_mask, batch_idx):
        import matplotlib.pyplot as plt
        import os
        for scene_idx in range(neg_seeds.shape[0]):
            if scene_idx > 32: break
            plt.figure(figsize=(10, 6))
            for person_index in range(neg_seeds.shape[1]):
                if neg_mask[scene_idx, person_index].sum() != neg_seeds.shape[-2]*2: continue
                # past GT
                neg_hist_ = neg_hist.cpu()[scene_idx, person_index, :, :]
                plt.plot(neg_hist_[:, 0], neg_hist_[:, 1], '-.', markersize = 2, linewidth=1, color='k', label=f'past')
                # fut GT
                neg_seeds_ = torch.cat((neg_hist.cpu()[scene_idx, person_index, -1:, :], neg_seeds.cpu()[scene_idx, person_index, :, :]), dim=0)
                plt.plot(neg_seeds_[:, 0], neg_seeds_[:, 1], '-.', markersize = 2, linewidth=1, color='b', label=f'y_GT')

            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'Forecasts batch {scene_idx}')
            plt.legend()
            plt.grid(True)
                # plt.show()
            folder_name = '/mnt/jaewoo4tb/predplan_ddpo/temp'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            # self.logger.add_figure(f'viz/ddpo_imgidx{batch_index}', plt.gcf(), self.global_step)    
            # self.logger.flush()
            plt.savefig(f'{folder_name}/dataloader_{self.split}_ddpo_batch:{batch_idx}_scene:{scene_idx}.png')
            plt.close() 


