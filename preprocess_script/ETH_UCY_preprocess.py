import torch
import os
import tqdm
import yaml
import pytorch_lightning as pl
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dotdict import dotdict
from utils.eth_utils import preprocess
from utils.eth_utils import get_ethucy_split
import copy

class ETH_data_saver(Dataset):
    def __init__(self, parser, split='train', frame_skip=1, data_root=None, output_dir=None, preprocessing_dataset_name=None):
        self.past_frames = parser['past_frames']
        self.min_past_frames = parser['min_past_frames']
        self.frame_skip = frame_skip
        self.split = split
        assert split in ['train', 'val', 'test'], 'error'
        seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
        self.init_frame = 0
        self.MAX_NUM_PERSON=150  # 원래 60

        process_func = preprocess
        self.data_root = data_root

        # print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        self.robot_indexes = []
        for seq_name in self.sequence_to_load:
            # print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, self.split)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        new_samp_list = []
        print('Dataloader generating valid sample indexes...')
        for sample_index in tqdm.tqdm(self.sample_list):
            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            data = seq(frame)
            if data is None: continue
            if len(data['pre_motion_3D']) == 1: continue       # If only one agent, skip
            for ii in range(len(data['pre_motion_3D'])):
                new_samp_list.append(sample_index)
                self.robot_indexes.append(ii)
        self.sample_list = new_samp_list
        assert len(self.robot_indexes) == len(self.sample_list)
        print(f'split: {split}, total num samples: {len(self.sample_list)}')
        print('Complete!')

        # NOTE: init에서 미리 처리
        for idx in tqdm.tqdm(range(len(self.sample_list))):
            sample_index = self.sample_list[idx]
            robot_index = self.robot_indexes[idx]
            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            data = seq(frame)
            
            pre_motion, fut_motion = data['pre_motion_3D'], data['fut_motion_3D']
            pre_motion = torch.stack(pre_motion)
            fut_motion = torch.stack(fut_motion)
            num_person_in_scene = pre_motion.shape[0] - 1
            
            robot_mask = torch.zeros(pre_motion.shape[0], dtype=bool)
            robot_mask[robot_index] = True
            human_mask = ~robot_mask
            pos_hist = pre_motion[robot_mask]
            pos_seeds = fut_motion[robot_mask]
            neg_hist = pre_motion[human_mask]
            neg_seeds = fut_motion[human_mask]
            
            # Calculate distance
            cur_pos_human, cur_pos_robot = neg_hist[:, -1], pos_hist[:, -1]
            diff_h2h = cur_pos_human.unsqueeze(0) - cur_pos_human.unsqueeze(1)
            diff_r2h = cur_pos_human.unsqueeze(0) - cur_pos_robot.unsqueeze(1)
            
            neg_hist_clone, pos_hist_clone, neg_seeds_clone, pos_seeds_clone = neg_hist.clone(), pos_hist.clone(), neg_seeds.clone(), pos_seeds.clone()
            # Human: To individual's local coordinates at t=0
            neg_seeds -= neg_hist_clone[:,-1].unsqueeze(1)
            neg_hist -= neg_hist_clone[:,-1].unsqueeze(1)
            
            # Robot: To local coordinates at t=0
            pos_seeds -= pos_hist_clone[:,-1].unsqueeze(1)
            pos_hist -= pos_hist_clone[:,-1].unsqueeze(1)
            
            neg_mask = torch.zeros(neg_seeds.shape)
            neg_mask[:num_person_in_scene,:,:] = 1
            neg_mask = neg_mask.bool()
            
            robot_states = pos_hist[:,-1]
            robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)
            human_states = neg_hist[:,-1]
            human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
            action = torch.zeros((1))
            pad_mask_person2d = torch.zeros((self.MAX_NUM_PERSON, self.MAX_NUM_PERSON))
            pad_mask_person2d[:num_person_in_scene, :num_person_in_scene] = 1
            pad_mask_person1d = torch.zeros((self.MAX_NUM_PERSON))
            pad_mask_person1d[:num_person_in_scene] = 1
            
            # Masked output
            human_states_, neg_seeds_, neg_hist_, neg_mask_ = torch.zeros((self.MAX_NUM_PERSON, 4)), torch.zeros((self.MAX_NUM_PERSON, neg_seeds.shape[1], 2)), torch.zeros((self.MAX_NUM_PERSON, neg_hist.shape[1], 2)), torch.zeros((self.MAX_NUM_PERSON, neg_seeds.shape[1], 2))
            human_states_[:num_person_in_scene,:], neg_seeds_[:num_person_in_scene,:,:], neg_hist_[:num_person_in_scene,:,:], neg_mask_[:num_person_in_scene,:,:] = human_states, neg_seeds, neg_hist, neg_mask
            diff_h2h_, diff_r2h_ = torch.zeros((self.MAX_NUM_PERSON, self.MAX_NUM_PERSON, 2)), torch.zeros((1, self.MAX_NUM_PERSON, 2))
            diff_h2h_[:num_person_in_scene, :num_person_in_scene, :], diff_r2h_[:, :num_person_in_scene, :] = diff_h2h, diff_r2h
            neg_hist_clone_ = torch.zeros((self.MAX_NUM_PERSON, neg_hist.shape[1], 2))
            neg_seeds_clone_ = torch.zeros((self.MAX_NUM_PERSON, neg_seeds.shape[1], 2))
            neg_hist_clone_[:num_person_in_scene,:,:] = neg_hist_clone
            neg_seeds_clone_[:num_person_in_scene,:,:] = neg_seeds_clone
            # self.visualization_function_dataset(neg_seeds_.unsqueeze(0), neg_hist_.unsqueeze(0), neg_mask_.unsqueeze(0).bool(), idx)
            
            # difference
            pos_seeds, neg_seeds_ = torch.cat((pos_hist[:,-1:], pos_seeds), dim=-2), torch.cat((neg_hist_[:,-1:], neg_seeds_), dim=-2)
            pos_seeds = pos_seeds[:, 1:] - pos_seeds[:, :-1]
            neg_seeds_ = neg_seeds_[:, 1:] - neg_seeds_[:, :-1]
            pos_hist = pos_hist[:, 1:] - pos_hist[:, :-1]
            neg_hist_ = neg_hist_[:, 1:] - neg_hist_[:, :-1]

            # NOTE: 필요한 부분만 사용
            neg_mask = neg_mask_.bool()
            pad_mask_person = pad_mask_person2d.bool()
            robot_dix = robot_index
            neg_hist, pos_hist, neg_seeds, pos_seeds = neg_hist_clone_, pos_hist_clone, neg_seeds_clone_, pos_seeds_clone

            # preprocessing
            _, H_T, _ = pos_hist.shape
            neg_seeds_mask = neg_mask
            neg_hist_mask = neg_mask[:,:H_T,:]
            agent_mask = neg_mask.sum(dim=(-2, -1)) > 0
            pos_hist = pos_hist.squeeze(0)
            pos_hist_mask = torch.ones_like(pos_hist, dtype=bool)
            pos_seeds_mask = torch.ones_like(pos_seeds, dtype=bool).squeeze(0)

            # NOTE: Ego Centric (Crowdnav와 동일하게 처리 가능)
            robot_center = pos_seeds[:,0,:]     # 1,2
            pos_seeds = pos_seeds - robot_center.unsqueeze(0)   # 1, 12, 2
            pos_hist = pos_hist - robot_center                  # 8, 2
            neg_seeds = neg_seeds - robot_center.unsqueeze(0)   # 5, 12, 2
            neg_hist = neg_hist - robot_center.unsqueeze(0)     # 5, 8, 2
            neg_seeds[~neg_seeds_mask] = 0
            neg_hist[~neg_hist_mask] = 0

            processed_data={
                "neg_seeds": neg_seeds,
                "neg_seeds_mask": neg_seeds_mask[:, :, 0:1],
                "neg_hist": neg_hist,
                "neg_hist_mask": neg_hist_mask[:, :, 0:1],
                "agent_mask": agent_mask,
                "pad_mask_person": pad_mask_person,
                "pos_seeds": pos_seeds,
                "pos_seeds_mask": pos_seeds_mask[:, 0:1],
                "pos_hist": pos_hist,
                "pos_hist_mask": pos_hist_mask[:, 0:1]
            }

            save_path = os.path.join(output_dir, f'ETH_UCY_processed/{preprocessing_dataset_name}/{split}/{idx}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(processed_data, save_path)

    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

if __name__ == "__main__":
    for preprocessing_dataset_name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    # preprocessing_dataset_name = "eth"      # TODO: change here!! eth, hotel, univ, zara1, zara2
        output_dir = "data_preprocessed"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"configs/baseline_gametheoretic_{preprocessing_dataset_name}.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    
        horizon, history, plan_horizon = cfg["model"]["horizon"], cfg["model"]["history"], cfg["model"]["horizon"]
        train_parser = {'split': 'train', 'past_frames': history, 'future_frames': horizon, 'min_past_frames': history, \
            'frame_skip': 1, 'data_root': '../data/eth_ucy', 'dataset': cfg["dataset"]["name"], 'min_future_frames': horizon, 'traj_scale': 1, \
                'load_map': False}
        train_parser = dotdict(train_parser)
        # breakpoint()
        ETH_data_saver(train_parser, split='train', data_root="data/eth_ucy", output_dir=output_dir, preprocessing_dataset_name=preprocessing_dataset_name)
        ETH_data_saver(train_parser, split='val', data_root="data/eth_ucy", output_dir=output_dir, preprocessing_dataset_name=preprocessing_dataset_name)


