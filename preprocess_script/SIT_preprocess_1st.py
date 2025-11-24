# ### JRDB parser with image captioning via llava-next ###
# v2: use video captioning instead of image captioning
from PIL import Image
import cv2
import numpy as np
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('..')
import os.path as osp
import torch
from torch import nn
import glob
from tqdm import tqdm
import argparse
import copy
import json
from preprocess_script.preprocess_utils.utils import *
from preprocess_script.preprocess_utils.plot_2d_3d import *
from parser_model.captions_sit_giwon import Caption3DPoseEngine
SCENE_NAMES = sorted(glob.glob('data/SIT_dataset/sit_new/*'))
USE_FRAME = 8
PARSE_SCENE = 3

def split_list(lst, k):
    # Calculate the size of each chunk
    avg = len(lst) // k
    remainder = len(lst) % k
    
    chunks = []
    start = 0
    for i in range(k):
        # Distribute the remainder across the first 'remainder' chunks
        chunk_size = avg + 1 if i < remainder else avg
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size
    
    return chunks

def main_parse(scene_idx):
    print("Start parsing.")
    VERSION = 'SIT_processed_1st'
    VLM_TYPE = 'vllm' # ['local', 'vllm']   for local, use bev_jw. for vllm, use textraj (conda env)
    VISUALIZE = False
    folder_path = "data_preprocessed"
    os.makedirs(folder_path, exist_ok=True)

    default_save_dir = 'data_preprocessed/' + VERSION
    base_dir = 'data/SIT_dataset/sit_new/'
    # scenes = sorted(glob.glob('images/image_0/*'))
    # SCENE_NAMES_split = split_list(SCENE_NAMES, split_nums)
    caption_engine = Caption3DPoseEngine('eval_sit', USE_FRAME)
    # breakpoint()
    for sceneIdx, scene_ in enumerate(SCENE_NAMES):     # 63개
        
        scene = os.path.basename(scene_)
        # if scene_ not in SCENE_NAMES_split[split_idx]: continue
        # if sceneIdx!=scene_idx: continue   # DEBUG
        # breakpoint()
        # Skip scene without annotation
        dir_items = glob.glob(scene_+'/*')
        dir_items = [os.path.basename(foo) for foo in dir_items]
        # breakpoint()
        if not ('cam_img' in dir_items and 'ego_trajectory' in dir_items and 'label_2d' in dir_items and 'label_3d' in dir_items):
            continue
        
        print(f'Processing scene: {scene}, {sceneIdx+1} out of {len(SCENE_NAMES)} scenes.')

        os.makedirs(default_save_dir, exist_ok=True)
        output_savedir = os.path.join(default_save_dir, scene)
        frame_save_dir = os.path.join(output_savedir, 'frames')
        cap_frame_save_dir = os.path.join(output_savedir, 'cap_frames')
        # os.makedirs(frame_save_dir, exist_ok=True)
        # os.makedirs(cap_frame_save_dir, exist_ok=True)
        save_data, save_interaction = {}, {}

        caption_engine.load_files(os.path.join(base_dir, scene, 'label_3d'), os.path.join(base_dir, scene, 'label_2d'), os.path.join(base_dir, scene, 'cam_img'), os.path.join(base_dir, scene, 'ego_trajectory'))
        num_frames = len(glob.glob(os.path.join(base_dir, scene, 'cam_img', '1', 'data_rgb')+'/*.png'))

        start_frame_idx = 0
        last_frame_idx = num_frames     # 200
        # breakpoint()
        for frame_idx in tqdm(range(start_frame_idx, num_frames)):
            # if frame_idx < 40: continue
            # frame_idx = 7
            # sceneIdx = 0
            # USE_FRAME = 8
            annot3d, ego_pos, ego_yaw = caption_engine.preprocess_frame(frame_idx, sceneIdx, USE_FRAME)     # process trajectory information, save cropped imgs
            # breakpoint()
            if frame_idx < start_frame_idx+USE_FRAME-1:   # gather frames   # 0,1,2,3,4,5,6 skip
                continue

            # breakpoint()
            # NOTE: video captioning은 사용 X
            # caption_engine.caption_multiframe(frame_idx, sceneIdx, USE_FRAME-1, cap_frame_save_dir)
            # caption_engine.regress_3dpose(frame_idx)
            
            # stopPrint(caption_engine.caption_single)
            # stopPrint(caption_engine.caption_dual)
            
            save_data[frame_idx] = {}
            # breakpoint()
            for agent_id, value in annot3d.items():
                # breakpoint()
                # if frame_idx not in caption_engine.agents[agent_id].global_pos.keys(): 
                #     breakpoint()
                #     continue
                # breakpoint()
                agent_position = value[:2]
                robot_pos = ego_pos
                robot_ori = ego_yaw
                rot_z = value[2]
                if agent_id not in save_data[frame_idx].keys():
                    save_data[frame_idx][agent_id] = {}
                        
                # save_data[frame_idx][agent_id]["description"] = caption_engine.agents[agent_id].caption
                save_data[frame_idx][agent_id]["local_position"] = agent_position - robot_pos
                save_data[frame_idx][agent_id]["global_position"] = agent_position
                # save_data[frame_idx][agent_id]["pose"] = caption_engine.agents[agent_id].pose
                save_data[frame_idx][agent_id]["robot_pos"] = robot_pos
                save_data[frame_idx][agent_id]["robot_ori"] = robot_ori
                save_data[frame_idx][agent_id]["rot_z"] = rot_z
                # save_data[frame_idx][agent_id]["cluster_id"] = -1   # default for agents with no cluster
            
            
            # visualize
            if VISUALIZE:
                if frame_idx < 30:
                    os.makedirs(frame_save_dir, exist_ok=True)
                    plot_3d_human(frame_idx, save_data, None, save_dir=frame_save_dir)
                # visualize_scene(frame_idx, caption_engine.img, caption_engine.label_2d, caption_engine.label_3d, person, interaction, os.path.join(frame_save_dir, f'{str(frame_idx).zfill(4)}.png'))
            # if frame_idx == 100: break
        # breakpoint()
        torch.save(save_data, os.path.join(default_save_dir, scene+f'_agents_{start_frame_idx}_to_{last_frame_idx}.pt'))
        # torch.save(save_interaction, os.path.join(default_save_dir, scene+f'_interactions_{start_frame_idx}_to_{last_frame_idx}.pt'))
        # break

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process split_nums and split_idx.")
    parser.add_argument('--scene_idx', type=int, default=0, help='Index of the current split')  # 0~62      # 47이 디버깅으로 굿

    args = parser.parse_args()
    scene_idx = args.scene_idx
    print(f'Start main parse: scene idx: {scene_idx}')
    main_parse(scene_idx)
    print(f'Finished! scene idx: {scene_idx}')