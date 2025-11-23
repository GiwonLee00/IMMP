import scipy.io
import numpy as np
import sys
from pathlib import Path
import torch
import tqdm
jerk_magnitude_list = []
import os

# 2.5Hz에 맞춰 리샘플링하기 위한 인덱스를 계산하는 함수
def resample_data(x, y, target_frequency_hz=2.5, original_frequency_hz=100):
    # 타겟 주파수에 맞는 시간 간격
    target_dt = 1 / target_frequency_hz
    original_dt = 1 / original_frequency_hz
    
    # 원본 데이터의 총 길이
    num_points = len(x)
    
    # 새로운 데이터의 개수 (2.5Hz에 맞춰 리샘플링)
    target_num_points = int(num_points * (target_frequency_hz / original_frequency_hz))
    
    # 2.5Hz에 맞게 샘플을 선택
    sampled_x = x[::int(original_frequency_hz / target_frequency_hz)]
    sampled_y = y[::int(original_frequency_hz / target_frequency_hz)]
    
    return sampled_x, sampled_y

number_of_skiped_data = 2121
human_position_total = []
robot_position_total = []
len_of_scenario = []
cur_idx = 0
for filename in [f'data/THOR/Exp_2_run_1.mat', f'data/THOR/Exp_2_run_2.mat', f'data/THOR/Exp_2_run_3.mat', f'data/THOR/Exp_2_run_4.mat', f'data/THOR/Exp_2_run_5.mat']:
    mat_data = scipy.io.loadmat(filename)
    key_list = mat_data.keys()

    filtered_keys = [key for key in key_list if 'run' in key][0]
    
    label_data = mat_data[filtered_keys][0][0][6][0][0][0][0][0][1]
    robot_indices = [i for i, item in enumerate(label_data[0]) if 'Citi' in item[0]]
    
    ego_agent_1to5_list = []
    surround_agent_1to5_list = []
    # total_ego_agent = torch.zeros(2, len(x_resampled))
    for robot_number in robot_indices:
        robot_trajectory = mat_data[filtered_keys][0][0][6][0][0][0][0][0][2][robot_number]
        x_pos = robot_trajectory[0]
        y_pos = robot_trajectory[1]

        # 2.5Hz로 리샘플링
        x_resampled, y_resampled = resample_data(x_pos, y_pos, target_frequency_hz=2.5, original_frequency_hz=100)
        # int(re.search(r'\d+$', str(label_data[0,robot_number])).group())
        current_scene = int(str(label_data[0,robot_number]).split(" - ")[1].strip("[]' "))
        # breakpoint()
        # for i in 
        #     label_data[0, i]


        helmet_info = [(i, int(str(item[0]).split("_")[1].split(" - ")[0]), int(str(item[0]).split(" - ")[1])) 
                    for i, item in enumerate(label_data[0])
                    if str(item[0]).startswith("Helmet") and int(str(item[0]).split(" - ")[1].strip("[]' ")) == current_scene]

        # breakpoint()
        total_surround_agent = torch.zeros(9, 2, len(x_resampled), dtype=torch.float32)  # max agent number(=9), 2, 28164
        for cur_helmet_data_indice, person_num, frame_num in helmet_info:
            # breakpoint()
            cur_surround_agent_x_pos = mat_data[filtered_keys][0][0][6][0][0][0][0][0][2][cur_helmet_data_indice][0]
            cur_surround_agent_y_pos = mat_data[filtered_keys][0][0][6][0][0][0][0][0][2][cur_helmet_data_indice][1]
            sur_agent_x_pos_resampled, sur_agent_y_pos_resampled = resample_data(cur_surround_agent_x_pos, cur_surround_agent_y_pos, target_frequency_hz=2.5, original_frequency_hz=100)
            # breakpoint()
            # if torch.tensor(sur_agent_x_pos_resampled).shape[0] != 705:
            #     breakpoint()
            # print("person_num: ", person_num-2)
            total_surround_agent[person_num-2, 0, :] = torch.tensor(sur_agent_x_pos_resampled, dtype=torch.float32)
            total_surround_agent[person_num-2, 1, :] = torch.tensor(sur_agent_y_pos_resampled, dtype=torch.float32)
            # breakpoint()
        surround_agent_1to5_list.append(total_surround_agent)
        robot_xy = np.vstack([x_resampled, y_resampled])
        # breakpoint()
        ego_agent_1to5_list.append(torch.tensor(robot_xy, dtype=torch.float32))
        len_of_scenario.append(cur_idx)
        cur_idx += torch.tensor(robot_xy, dtype=torch.float32).shape[-1]
        # TODO: 로봇 경로도 담아주기. 
        # 사람들 nan 있는데, 마스크 싀워주기

    # surround_agent_1to5_list: 5개의 len, 9,2,705사이즈의 텐서    9는 surround agent를 의미
    # ego_agent_1to5_list: 5개의 len, 2,705사이즈의 텐서
    # 
    human_position = torch.cat(surround_agent_1to5_list, dim=-1)        # 9, 2, 705*5
    robot_position = torch.cat(ego_agent_1to5_list, dim=-1)             # 2, 705*5

    human_position_total.append(human_position)
    robot_position_total.append(robot_position)

human_data = torch.cat(human_position_total, dim=-1)        # 9, 2, 705*5 + ...
robot_data = torch.cat(robot_position_total, dim=-1)        # 2, 705*5 + ...


# len_of_scenario -> start idx list이다.
idx_start = len_of_scenario 
cur_count = 0
history = 8
horizon = 12
MAX_NUM_PERSON = 150
output_dir = "data_preprocessed"
# new_value_data = []
neg_hist = []
pos_hist = []
neg_state = []
pos_state = []

for start_idx, end_idx in zip(idx_start[:-1], idx_start[1:]):
    for idx in range(start_idx + history, end_idx - horizon):
        human_future = human_data[:, :, idx:idx+horizon]
        robot_future = robot_data[:, idx:idx+horizon]

        human_history = human_data[:, :, idx-history:idx] # 0:8
        robot_history = robot_data[:, idx-history:idx]

        # human_history = torch.roll(human_history, -1, 0)
        # human_history[-1] = human_data[idx, :, :2]
        
        # robot_history = torch.roll(robot_history, -1, 0)
        # robot_history[-1] = robot_data[idx, :2]       
        
        # new_robot_data[cur_count] = robot_data[idx]
        # new_human_data[cur_count] = human_data[idx]
        # new_action_data[cur_count] = action_data[idx]
        # new_value_data[cur_count] = value_data[idx]
        
        neg_hist.append(human_history)
        pos_hist.append(robot_history)
        neg_state.append(human_future)
        pos_state.append(robot_future)
        # breakpoint()

neg_hist = torch.stack(neg_hist)
pos_hist = torch.stack(pos_hist)
neg_state = torch.stack(neg_state)
pos_state = torch.stack(pos_state)
        
# TODO: mask생성하기
neg_hist_mask_total = ~torch.isnan(neg_hist)   
pos_hist_mask_total = ~torch.isnan(pos_hist)   
neg_seeds_mask_total = ~torch.isnan(neg_state)   
pos_seeds_mask_total = ~torch.isnan(pos_state)   
# breakpoint()
skip_data = 0
for idx in tqdm.tqdm(range(pos_state.shape[0])):
    pos_seeds = pos_state[idx].permute(1, 0).unsqueeze(0)
    neg_seeds = neg_state[idx].permute(0, 2, 1)
    pos_hist_sample = pos_hist[idx].permute(1, 0)
    neg_hist_sample = neg_hist[idx].permute(0, 2, 1)
    neg_hist_mask = neg_hist_mask_total[idx].permute(0, 2, 1)
    pos_hist_mask = pos_hist_mask_total[idx].permute(1,0)
    neg_seeds_mask = neg_seeds_mask_total[idx].permute(0, 2, 1)
    pos_seeds_mask = pos_seeds_mask_total[idx].permute(1,0).unsqueeze(0)
    # breakpoint()

    # MAX person version
    num_person_in_scene = 9  # TODO
    # breakpoint()
    neg_seeds = torch.cat([
        neg_seeds,
        torch.zeros(MAX_NUM_PERSON - num_person_in_scene, horizon, 2, dtype=torch.float32)
    ], dim=0)
    neg_seeds_mask = torch.cat([
        neg_seeds_mask,
        # torch.ones(num_person_in_scene, horizon, 2, dtype=torch.bool),      # TODO: one 대신 nan 값 마스킹 한거 드갓ㅁ.
        torch.zeros(MAX_NUM_PERSON - num_person_in_scene, horizon, 2, dtype=torch.bool)
    ], dim=0)
    neg_hist_sample = torch.cat([
        neg_hist_sample,
        torch.zeros(MAX_NUM_PERSON - num_person_in_scene, history, 2, dtype=torch.float32)
    ], dim=0)
    neg_hist_mask = torch.cat([
        neg_hist_mask,
        # torch.ones(num_person_in_scene, history, 2, dtype=torch.bool),
        torch.zeros(MAX_NUM_PERSON - num_person_in_scene, history, 2, dtype=torch.bool)
    ], dim=0)
    # breakpoint()

    # pos_hist_mask = torch.ones_like(pos_hist_sample, dtype=bool)
    # pos_seeds_mask = torch.ones_like(pos_seeds, dtype=bool).squeeze(0)
    # pos_seeds[~pos_seeds_mask] = 0
    if torch.isnan(pos_seeds).any():
        # print("continue for center point is zero")
        skip_data += 1
        continue
    '''
    if torch.isnan(pos_seeds[:, 0, :]).any():
        print("pos_seeds: ", pos_seeds)
        print("continue for center point is zero")
        skip_data += 1
        continue
        breakpoint()
        raise ValueError("Error: Tensor contains NaN values.")
    if torch.isnan(agent_mask).any():
        breakpoint()
        raise ValueError("Error: Tensor contains NaN values.")
    if torch.isnan(pad_mask_person).any():
        breakpoint()
        raise ValueError("Error: Tensor contains NaN values.")
    # if torch.isnan(neg_hist_sample).any():
    #     breakpoint()
    #     raise ValueError("Error: Tensor contains NaN values.")
    '''
    # Ego Centric
    robot_center = pos_seeds[:, 0, :]  # 1,2
    pos_seeds = pos_seeds - robot_center.unsqueeze(0)  # 1, 12, 2
    pos_hist_sample = pos_hist_sample - robot_center  # 8, 2
    neg_seeds = neg_seeds - robot_center.unsqueeze(0)  # 5, 12, 2
    neg_hist_sample = neg_hist_sample - robot_center.unsqueeze(0)  # 5, 8, 2
    # breakpoint()
    neg_seeds[~neg_seeds_mask] = 0
    neg_hist_sample[~neg_hist_mask] = 0
    pos_seeds[~pos_seeds_mask] = 0
    pos_hist_sample[~pos_hist_mask] = 0
    # breakpoint()
    valid_agent_mask = neg_hist_mask[:, -1, :] != 0     # t = 0 일 때 
    valid_agent_mask = valid_agent_mask.unsqueeze(1)  
    valid_agent_mask_his = valid_agent_mask.expand(-1, history, -1)
    valid_agent_mask_fut = valid_agent_mask.expand(-1, horizon, -1)
    # breakpoint()
    neg_seeds[~valid_agent_mask_fut] = 0
    neg_seeds_mask[~valid_agent_mask_fut] = 0
    neg_hist_sample[~valid_agent_mask_his] = 0
    neg_hist_mask[~valid_agent_mask_his] = 0
    # breakpoint()

    if torch.sum(neg_seeds) == 0:  # 주변 agent가 하나도 없으면 pass
        skip_data += 1
        continue

    total_timestep_human_mask = torch.cat([neg_seeds_mask, neg_hist_mask], dim=1)
    # breakpoint()
    agent_mask = total_timestep_human_mask.sum(dim=(-2, -1)) > 0
    
    # TODO: 그냥 1 채우면 안됨. SIT, JRDB 방식을 참고.
    # pad_mask_person = torch.zeros((MAX_NUM_PERSON, MAX_NUM_PERSON), dtype=torch.bool)
    # pad_mask_person[:num_person_in_scene, :num_person_in_scene] = 1     
    pad_mask_person = agent_mask.unsqueeze(0) & agent_mask.unsqueeze(1)  # [max_agent, max_agent]    

    processed_data={        # / 1000  : mm to m
        "neg_seeds": neg_seeds / 1000,
        "neg_seeds_mask": neg_seeds_mask[:, :, 0:1],
        "neg_hist": neg_hist_sample / 1000,
        "neg_hist_mask": neg_hist_mask[:, :, 0:1],
        "agent_mask": agent_mask,
        "pad_mask_person": pad_mask_person,
        "pos_seeds": pos_seeds / 1000,
        "pos_seeds_mask": pos_seeds_mask[0, :, 0:1],
        "pos_hist": pos_hist_sample / 1000,
        "pos_hist_mask": pos_hist_mask[:, 0:1]
    }
    # if idx == 105:
    #     breakpoint()
    if idx - skip_data < int((pos_state.shape[0]-number_of_skiped_data) / 2):   # 8235
        save_path = os.path.join(output_dir, f'THOR_processed/train/{idx-skip_data}.pt')
        final_idx = idx-skip_data
    else: 
        save_path = os.path.join(output_dir, f'THOR_processed/valid/{idx - final_idx - 1 - skip_data}.pt') 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(processed_data, save_path)

# breakpoint()
print("skip data: ", skip_data)
