import torch
import os
from tqdm import tqdm
import yaml
import pytorch_lightning as pl
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 기본 경로 설정
root_dir = "data_preprocessed/SIT_processed_2nd"
output_dir = "data_preprocessed/SIT_processed_3rd"

# 저장할 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 모든 폴더 탐색
folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
# folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if not os.path.isdir(os.path.join(root_dir, folder))]
# folders = [root_dir]
# breakpoint()
def preprocess_data(data, cfg):
    max_agent = cfg['model'].get('max_agent', 150)
    history = cfg['model']['history']
    ego_centric = cfg["dataset"]["ego_centric"]
    # Extract data
    agent_positions = data["positions"]
    padding_mask = data["padding_mask"]
    robot_pos = data["robot_pos"].squeeze(0)
    robot_mask = ~data["robot_mask"].squeeze(0)

    # Apply padding and masks
    padded_agent, agent_path_mask, agent_mask = pad_create_mask(agent_positions, padding_mask, max_agent)
    agent_2D_mask = agent_mask.unsqueeze(0) & agent_mask.unsqueeze(1)  # [max_agent, max_agent]

    if ego_centric:
        robot_pos_current = robot_pos[history:history + 1, :]  # Starting point of robot
        robot_pos = robot_pos - robot_pos_current
        padded_agent = padded_agent - robot_pos_current.unsqueeze(1)

    # Preprocess data
    processed_data = {
        "neg_seeds": padded_agent[:, history:, :],  # 50, 12, 2
        "neg_seeds_mask": agent_path_mask[:, history:].unsqueeze(-1),  # 50, 12, 1
        "neg_hist": padded_agent[:, :history, :],  # 50, 8, 2
        "neg_hist_mask": agent_path_mask[:, :history].unsqueeze(-1),  # 50, 8, 1
        "agent_mask": agent_mask,
        "pad_mask_person": agent_2D_mask,
        "pos_seeds": robot_pos[history:, :].unsqueeze(0),
        "pos_seeds_mask": robot_mask[history:].unsqueeze(-1),
        "pos_hist": robot_pos[:history, :],
        "pos_hist_mask": robot_mask[:history].unsqueeze(-1),
    }

    return processed_data 

def pad_create_mask(agent_positions, padding_mask, max_agent=150):
    feature_dim1, feature_dim2 = agent_positions.size(1), agent_positions.size(2)

    padded_agent = torch.zeros((max_agent, feature_dim1, feature_dim2))  # [max_agent, history, 2]
    agent_path_mask = torch.ones((max_agent, feature_dim1), dtype=torch.bool)
    agent_mask = torch.ones((max_agent), dtype=torch.bool)
    
    agent_count = agent_positions.size(0)  # Current number of agents
    if agent_count > max_agent:
        raise ValueError("Increase MAX_AGENT, current count exceeds limit!")
    else:
        padded_agent[:agent_count, :, :] = agent_positions
        agent_path_mask[:agent_count, :] = padding_mask
        agent_mask[:agent_count] = False
    
    return padded_agent, ~agent_path_mask, ~agent_mask  # Valid positions have True

if __name__ == "__main__":
    with open("configs/baseline_gametheoretic_SIT.yaml", "r") as f:
        cfg = yaml.safe_load(f)
   
    for folder in tqdm(folders, desc="Processing Folders"):
        pt_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")]
        # breakpoint()
        for pt_file in tqdm(pt_files, desc=f"Processing {folder}", leave=False):
            
            data = torch.load(pt_file)
            
            processed_data = preprocess_data(data, cfg)
            
            relative_path = os.path.relpath(pt_file, root_dir)
            save_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(processed_data, save_path)

    print("Preprocessing 완료!")