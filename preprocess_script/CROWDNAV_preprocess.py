import torch
import os
import tqdm
import yaml
import pytorch_lightning as pl
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dotdict import dotdict
import copy

class CROWDNAV_data_saver(Dataset):
    def __init__(self, data_root, vmin=0.0, 
    horizon=12, history=8, plan_horizon=12, output_dir=None):
        '''
        # Assumes planning horizon <= horizon
        '''
        data = torch.load(data_root)
        robot_data = data[0]
        human_data = data[1]
        action_data = data[2]
        value_data = data[3]
        len = 340869
        pos_seq = torch.zeros(len, horizon, 2)
        neg_seq = torch.zeros(len, horizon, human_data.size(1), 2)
        new_robot_data = torch.zeros(len, 6)
        new_human_data = torch.zeros(len, human_data.size(1), 4)
        new_action_data = torch.zeros(len, 2)
        # breakpoint()
        new_value_data = torch.zeros(len,1)
        # neg_mask = torch.ones(human_data.size(0), horizon, human_data.size(1), 2)
        
        pos_hist = torch.zeros(len, history, 2)
        neg_hist = torch.zeros(len, history, human_data.size(1), 2)
        
        num_humans = human_data.size(1)
        
        # remove samples at the end of episodes
        vx = robot_data[1:, 2]
        dx = robot_data[1:, 0] - robot_data[:-1, 0]
        diff = dx - vx * 0.25
        idx_done = (diff.abs() > 1e-6).nonzero(as_tuple=False)
        
        start_state = torch.Tensor([0, -4, 0.0, 0.0, 0, 4.0])
        idx_start = (torch.linalg.norm(robot_data-start_state, dim=1) < 1e-5).nonzero(as_tuple=False)
        
        idx_start = [int(x) for x in idx_start[:, 0]]
        idx_start.append(robot_data.size(0))
        
        
        cur_count = 0
        # new_value_data = []
        for start_idx, end_idx in zip(idx_start[:-1], idx_start[1:]):
            # print(human_data[start_idx, :, :2].shape)
            human_history = human_data[start_idx, :, :2].repeat(history, 1, 1)   
            robot_history = robot_data[start_idx, :2].repeat(history, 1)           # 처음 start idx에 있는 로봇의 상태 가져옴. 이걸 동일하게 repeat해서 8개 만큼 만들어줌.
            
            for idx in range(start_idx, end_idx - horizon):
                human_future = human_data[idx+1:idx+horizon+1, :, :2]
                robot_future = robot_data[idx+1:idx+horizon+1, :2]
       
                human_history = torch.roll(human_history, -1, 0)
                human_history[-1] = human_data[idx, :, :2]
                
                robot_history = torch.roll(robot_history, -1, 0)
                robot_history[-1] = robot_data[idx, :2]       
                
                new_robot_data[cur_count] = robot_data[idx]
                new_human_data[cur_count] = human_data[idx]
                new_action_data[cur_count] = action_data[idx]
                new_value_data[cur_count] = value_data[idx]
                
                neg_hist[cur_count] = human_history
                pos_hist[cur_count] = robot_history
                neg_seq[cur_count] = human_future
                pos_seq[cur_count] = robot_future
                
                # new_value_data.append(value_data[idx])
                cur_count += 1
        # breakpoint()
        # remove bad experience for imitation
        mask = (new_value_data > vmin).squeeze()
        pos_seq = pos_seq[:, :plan_horizon, :]

        self.robot_state = new_robot_data[mask]
        self.human_state = new_human_data[mask]
        self.action_target = new_action_data[mask]
        self.pos_state = pos_seq[mask]
        self.neg_state = neg_seq[mask]
        self.pos_hist = pos_hist[mask]
        self.neg_hist = neg_hist[mask]

        self.MAX_NUM_PERSON = 150
        self.history = history
        self.horizon = horizon

        # TODO: 미리 연산해두도록 함.
        # breakpoint()
        # self.pos_state.shape[0]
        for idx in tqdm.tqdm(range(self.pos_state.shape[0])):
            pos_seeds = self.pos_state[idx].unsqueeze(0)
            neg_seeds = self.neg_state[idx].permute(1, 0, 2)
            pos_hist_sample = self.pos_hist[idx]
            neg_hist_sample = self.neg_hist[idx].permute(1, 0, 2)

            # MAX person version
            num_person_in_scene = 5
            neg_seeds = torch.cat([
                neg_seeds,
                torch.zeros(self.MAX_NUM_PERSON - num_person_in_scene, horizon, 2)
            ], dim=0)
            neg_seeds_mask = torch.cat([
                torch.ones(num_person_in_scene, horizon, 2, dtype=torch.bool),
                torch.zeros(self.MAX_NUM_PERSON - num_person_in_scene, horizon, 2, dtype=torch.bool)
            ], dim=0)
            neg_hist_sample = torch.cat([
                neg_hist_sample,
                torch.zeros(self.MAX_NUM_PERSON - num_person_in_scene, self.history, 2)
            ], dim=0)
            neg_hist_mask = torch.cat([
                torch.ones(num_person_in_scene, self.history, 2, dtype=torch.bool),
                torch.zeros(self.MAX_NUM_PERSON - num_person_in_scene, self.history, 2, dtype=torch.bool)
            ], dim=0)
            agent_mask = neg_seeds_mask.sum(dim=(-2, -1)) > 0
            pad_mask_person = torch.zeros((self.MAX_NUM_PERSON, self.MAX_NUM_PERSON), dtype=torch.bool)
            pad_mask_person[:num_person_in_scene, :num_person_in_scene] = 1

            pos_hist_mask = torch.ones_like(pos_hist_sample, dtype=bool)
            pos_seeds_mask = torch.ones_like(pos_seeds, dtype=bool).squeeze(0)

            # Ego Centric
            robot_center = pos_seeds[:, 0, :]  # 1,2
            pos_seeds = pos_seeds - robot_center.unsqueeze(0)  # 1, 12, 2
            pos_hist_sample = pos_hist_sample - robot_center  # 8, 2
            neg_seeds = neg_seeds - robot_center.unsqueeze(0)  # 5, 12, 2
            neg_hist_sample = neg_hist_sample - robot_center.unsqueeze(0)  # 5, 8, 2
            neg_seeds[~neg_seeds_mask] = 0
            neg_hist_sample[~neg_hist_mask] = 0

            processed_data={
                "neg_seeds": neg_seeds,
                "neg_seeds_mask": neg_seeds_mask[:, :, 0:1],
                "neg_hist": neg_hist_sample,
                "neg_hist_mask": neg_hist_mask[:, :, 0:1],
                "agent_mask": agent_mask,
                "pad_mask_person": pad_mask_person,
                "pos_seeds": pos_seeds,
                "pos_seeds_mask": pos_seeds_mask[:, 0:1],
                "pos_hist": pos_hist_sample,
                "pos_hist_mask": pos_hist_mask[:, 0:1]
            }
            if idx < self.pos_state.shape[0] / 2:
                save_path = os.path.join(output_dir, f'CROWDNAV_processed/train/{idx}.pt')
            else: 
                save_path = os.path.join(output_dir, f'CROWDNAV_processed/valid/{idx - int(self.pos_state.shape[0] / 2)}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(processed_data, save_path)

if __name__ == "__main__":
    output_dir = "data_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"configs/baseline_gametheoretic_crowd_nav.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    horizon, history, plan_horizon = cfg["model"]["horizon"], cfg["model"]["history"], cfg["model"]["horizon"]
    CROWDNAV_data_saver("data/demonstration/data_imit.pt", horizon=horizon, history=history, plan_horizon=plan_horizon, output_dir=output_dir)


