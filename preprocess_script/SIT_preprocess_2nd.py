
"""
v2: only by number of agents in this scene, include robot position
- scene: {
    agent_position: 25 X 20 X 2
    agent_pose: 25 X 20 X 72
    agent_pose_mask: 25 X 20
    agent_description: 25 X 20 X embed_dim
    agent_description_mask: 25 X 20
    agent_mask: 25 X 20 => True 면 mask 된 정보

    object_position: N X 20 X 3 => xy, object_index
    object_mask: N X 20

    robot_position: 20 X 2
    robot_orientation: 20 X 1

    cluster_ID: 25 X 20 ⇒ cluster ID int
    cluster_embed: max(cluster ID) X 20 X embed_dim
    cluster_embed_mask: max(cluster ID) X 20

    dual_interaction: 25 X 25 X 20 X embed_dim
    dual_interaction_mask: 25 X 25 X 20
    }
"""
import argparse
import torch
from tqdm import tqdm
from preprocess_script.preprocess_utils.utils import *
# from parser_model.text_encoder import TextEncoder
# from parser_model.text_encoder_bert import TextEncoder_BERT, TextEncoder_TinyBERT
from itertools import permutations
from itertools import product
from scipy.spatial.transform import Rotation as R_
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from preprocess_script.preprocess_utils.utils_hivt import TemporalData



class PreProcess:
    def __init__(self, agent_data: dict, scene_name: str, scene_idx: int, object_data: dict=None):
        self.TEXT_EMBED_DIM = 312 # TinyBERT: 312, BERT:768
        self.MAX_OBJECT_NUM = 20
        self.MAX_CLUSTER_NUM = 25
        self.FRAME_LENGTH = 20
        self.FRAME_STEP = 4
        self.PAST_FRAME = 8
        self.FUT_FRAME = 12
        self.MAX_NUM_AGENTS = 40
        self.TEMP_NUM_AGENTS = 300
        self.scene_name = scene_name
        self.scene_idx = scene_idx

        self.agent_data = agent_data
        self.object_data = object_data
        
    def __call__(self, initial_frame, text_encoder):
        # print(f"start preprocessing from frame {initial_frame} to {initial_frame + (self.FRAME_LENGTH - 1) * self.FRAME_STEP}")
        scene = {}
        # if self.scene_name != "ocessed_1st/sit_v1_debug/Lobby_7-004_agents_0_to_200":
        #     return None, None, None,
        
        agent_position = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 2))
        agent_ids = torch.zeros((self.TEMP_NUM_AGENTS), dtype=torch.int) -1
        robot_position = torch.zeros(size=(1, self.FRAME_LENGTH, 2))
        agent_pose = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 72))
        agent_pose_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        agent_description = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, self.TEXT_EMBED_DIM))
        agent_description_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        agent_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        robot_mask = torch.ones(size=(1, self.FRAME_LENGTH), dtype=torch.bool)
        # dual_interaction = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, self.TEXT_EMBED_DIM))
        # dual_interaction_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        b_ori = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 2))

        encoded_text_emb = {}

        agent_id_to_idx = {}

        frames = torch.IntTensor(size=(self.FRAME_LENGTH, ))
        # breakpoint()
        # initial_frame => 0
        # self.FRAME_LENGTH => 20
        # self.FRAME_STEP => 4
        # frames => 20크기의 텐서
        # total_agentid_23 = 0
        for frame in range(self.FRAME_LENGTH):      
            actual_frame = initial_frame + frame * self.FRAME_STEP
            frames[frame] = actual_frame # 0, 4, 8, 12, ... 가 들어감.


            # self.agent_data.keys() -> 7~199까지...
            if actual_frame not in self.agent_data.keys(): continue
            # TODO: 여기서 첫 두 frame이 없다고 날라감. (0, 4를 무시해버림.) 중요!!!!
            # breakpoint()
            # if initial_frame == 120:
            #     if frame == 12:  # actual_frame = 148
            #         breakpoint()
            for agent_id, value in self.agent_data[actual_frame].items():

                # if actual_frame == 150:
                #     breakpoint()
                # if agent_id == 23:
                
                #     total_agentid_23 += 1
                #     breakpoint()
                robot_position[0, frame] = torch.tensor(value['robot_pos'][:2])
                robot_mask[0, frame] = False
                if agent_id not in agent_id_to_idx:
                    agent_id_to_idx[agent_id] = len(agent_id_to_idx) 
                
                agent_save_idx = agent_id_to_idx[agent_id]
                # 18번 agent가 0번 agent로 들어감.
                # if marker == False:
                # if torch.sum(agent_position[agent_save_idx, frame, :]) != 0:
                #     breakpoint()
                # marker = False
                # if agent_save_idx
                # if marker == True:
                #     if torch.sum(agent_position[agent_save_idx, frame, :]) != 0:
                #         breakpoint()
                # marker = False
                # breakpoint()
                agent_position[agent_save_idx, frame, :] = torch.from_numpy(np.array(value["global_position"][:2]))
                agent_mask[agent_save_idx, frame] = False       # mask 추가
                if agent_ids[agent_save_idx] == -1: agent_ids[agent_save_idx] = agent_id
                else: assert agent_ids[agent_save_idx] == agent_id

                # if value["pose"] is None:
                #     agent_pose[agent_save_idx, frame, :] = torch.zeros(size=(72,))
                # else:
                #     agent_pose[agent_save_idx, frame, :] = torch.from_numpy(value["pose"]["smpl_thetas"])
                #     b_ori[agent_save_idx, frame, :] = value["pose"]["b_ori"].cpu()[0,:2]
                #     agent_pose_mask[agent_save_idx, frame] = False
                
                '''
                # NOTE: 내가 날림.
                if value["description"] is not None:
                    if value["description"] not in encoded_text_emb.keys():
                        encoded_text_emb[value["description"]] = text_encoder(value["description"])
                    agent_description[agent_save_idx, frame, :] = encoded_text_emb[value["description"]]
                    agent_description_mask[agent_save_idx, frame] = False
                '''

        # breakpoint()
        # agent_position
            # check if all frame have 3 agents with pose
            # if torch.sum(~agent_pose_mask[:, frame]) > 2:
            #     # print(~agent_pose_mask[:, frame])
            #     print(f"Invalid data frames: only {torch.sum(~agent_pose_mask[frame])} pose data at frame {actual_frame}")
            #     return
        agent_mask_true = (~agent_mask[:,:self.PAST_FRAME]).sum(-1) > (self.PAST_FRAME//2)      # Enough frames for past (True is valid data)
        num_valid_agents = agent_mask_true.sum()
        if num_valid_agents < 1: return None, None, None            # 
        x, positions = agent_position.clone(), agent_position.clone()
        
        # padding_mask_all = torch.ones((self.MAX_NUM_AGENTS, T)).bool()
        # padding_mask = raw_file['padding']
        padding_mask_all = agent_mask
        
        bos_mask = torch.zeros(self.TEMP_NUM_AGENTS, self.PAST_FRAME, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask_all[:, 0]
        bos_mask[:, 1: self.PAST_FRAME] = padding_mask_all[:, : self.PAST_FRAME-1] & ~padding_mask_all[:, 1: self.PAST_FRAME]
        
        rotate_angles = torch.zeros(self.TEMP_NUM_AGENTS, dtype=torch.float)
        valid_indices = [torch.nonzero(row[:self.PAST_FRAME], as_tuple=True)[0] for row in ~padding_mask_all]
        valid_pose_indices = [torch.nonzero(row[:self.PAST_FRAME], as_tuple=True)[0] for row in ~agent_pose_mask]
        
        for actor_id in range(self.TEMP_NUM_AGENTS):
            if len(valid_indices[actor_id]) < 2: continue   # If zero or one frame of observation, skip
            if (valid_indices[actor_id][1:]-valid_indices[actor_id][:-1]).min()>1: continue # If no consecutive frames, skip
            if len(valid_pose_indices[actor_id]) > 0 and valid_pose_indices[actor_id].max() > self.PAST_FRAME-3:
                heading_vector = b_ori[actor_id, valid_pose_indices[actor_id].max()]
                rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0])
            else:
                heading_vector = x[actor_id, valid_indices[actor_id][-1]] - x[actor_id, valid_indices[actor_id][-2]]
                rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0]) 
                
        x[:,self.PAST_FRAME:] = x[:,self.PAST_FRAME:] - x[:,self.PAST_FRAME-1].unsqueeze(-2)
        x[:,1:self.PAST_FRAME] = x[:,1:self.PAST_FRAME] - x[:,:self.PAST_FRAME-1]
        x[:,0] = torch.zeros(x.shape[0], 2)
        padding_mask_all_temp = padding_mask_all.clone()
        padding_mask_all_temp[:,1:self.PAST_FRAME] = padding_mask_all[:,1:self.PAST_FRAME] | padding_mask_all[:,:self.PAST_FRAME-1]
        x[padding_mask_all_temp] = 0
        y = x[:,self.PAST_FRAME:]
        
        rotate_mat = torch.empty(self.TEMP_NUM_AGENTS, 3, 3)
        sin_vals = torch.sin(rotate_angles)
        cos_vals = torch.cos(rotate_angles)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals      # original: -
        rotate_mat[:, 0, 2] = 0
        rotate_mat[:, 1, 0] = sin_vals     # original: +
        rotate_mat[:, 1, 1] = cos_vals
        rotate_mat[:, 1, 2] = 0
        rotate_mat[:, 2, 0] = 0
        rotate_mat[:, 2, 1] = 0
        rotate_mat[:, 2, 2] = 1
        if y is not None:
            y = torch.bmm(y, rotate_mat[:, :2, :2]) 
        
        rotate_mat_ = torch.empty(num_valid_agents, 3, 3)
        agent_ids_ = torch.zeros((num_valid_agents), dtype=torch.int)
        positions_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        x_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        y_ = torch.Tensor(size=(num_valid_agents, self.FUT_FRAME, 2))
        agent_pose_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 72))
        agent_pose_mask_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        agent_description_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, self.TEXT_EMBED_DIM))
        agent_description_mask_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        padding_mask_all_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        dual_interaction_ = torch.Tensor(size=(num_valid_agents, num_valid_agents, self.FRAME_LENGTH, self.TEXT_EMBED_DIM))
        dual_interaction_mask_ = torch.ones(size=(num_valid_agents, num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        bos_mask_ = torch.zeros(num_valid_agents, self.PAST_FRAME, dtype=torch.bool)
        rotate_angles_ = torch.zeros(num_valid_agents, dtype=torch.float)
        
        rotate_mat_[:agent_mask_true.sum()] = rotate_mat[agent_mask_true]
        agent_ids_[:agent_mask_true.sum()] = agent_ids[agent_mask_true]
        rotate_angles_[:agent_mask_true.sum()] = rotate_angles[agent_mask_true]
        positions_[:agent_mask_true.sum()] = positions[agent_mask_true]
        # agent_position_[:agent_mask_true.sum()] = agent_position[agent_mask_true]
        agent_pose_[:agent_mask_true.sum()] = agent_pose[agent_mask_true]
        agent_pose_mask_[:agent_mask_true.sum()] = agent_pose_mask[agent_mask_true]
        agent_description_[:agent_mask_true.sum()] = agent_description[agent_mask_true]
        agent_description_mask_[:agent_mask_true.sum()] = agent_description_mask[agent_mask_true]
        padding_mask_all_[:agent_mask_true.sum()] = padding_mask_all[agent_mask_true]
        bos_mask_[:agent_mask_true.sum()] = bos_mask[agent_mask_true]
        x_[:agent_mask_true.sum()] = x[agent_mask_true]
        y_[:agent_mask_true.sum()] = y[agent_mask_true]
        
        edge_index = torch.LongTensor(list(permutations(range(num_valid_agents), 2))).t().contiguous()
        # if agent_mask_true.sum() == 43:
        #     
        # breakpoint()
        scene = {
                # "frames": frames, 
                'num_nodes': agent_mask_true.sum(),
                'rotate_mat': rotate_mat_,
                'agent_ids': agent_ids_,
                'scene': self.scene_name,
                'x': x_[:,:self.PAST_FRAME],
                'x_pose': agent_pose_[:, :self.PAST_FRAME],
                'x_pose_mask': agent_pose_mask_[:, :self.PAST_FRAME],
                'x_text': agent_description_[:, :self.PAST_FRAME],
                'x_text_mask': agent_description_mask_[:, :self.PAST_FRAME],
                'positions': positions_,
                'rotate_angles': rotate_angles_,
                'padding_mask': padding_mask_all_, # position masking
                'edge_index': edge_index,
                'bos_mask': bos_mask_,
                # 'bos_pose_mask': bos_pose_mask,
                'y': y_,
                'y_pose': agent_pose_[:, self.PAST_FRAME:],
                'y_pose_mask': agent_pose_mask_[:, self.PAST_FRAME:],
                'y_text': agent_description_[:, self.PAST_FRAME:],
                'y_text_mask': agent_description_mask_[:, self.PAST_FRAME:],
                'robot_pos': robot_position,
                'robot_mask': robot_mask,
                'scene_idx': self.scene_idx,
                'initial_frame': initial_frame
                # 'interaction': dual_interaction[:, :, self.PAST_FRAME-1, :],
                # 'interaction_mask': dual_interaction_mask[:, :, self.PAST_FRAME-1],
                # 'agent_description': agent_description[:, self.PAST_FRAME-1, :],
                # 'agent_description_mask': agent_description_mask[:, self.PAST_FRAME-1],
                # 'dual_interaction_mask': dual_interaction_mask,
                }

        data = TemporalData(**scene)
        return data, scene, (agent_mask_true).sum()
        # return scene, (agent_mask_true).sum()
        
    def rotate_root(pose, rotation_angle_degrees, axis):
        # Convert the angle to radians
        if torch.is_tensor(rotation_angle_degrees): 
            if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
        pose = pose.squeeze()
        # rotation_angle_radians = np.radians(rotation_angle_degrees)
        rotation_angle_radians = rotation_angle_degrees
        rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
        root_rotation_matrix = R_.from_rotvec(pose[:, :3]).as_matrix()
        new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
        new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
        new_pose = np.copy(pose)
        new_pose[:, :3] = new_root_rotation_vector
        return np.expand_dims(new_pose, 0)

if __name__ == "__main__":
    from glob import glob
    parser = argparse.ArgumentParser(description="Preprocessing script for dataset.")
    
    parser.add_argument("--scene_idx_start", type=int, default=0,
                        help="Scene idx out of 62 to start.")
    parser.add_argument("--scene_idx_end", type=int, default=62,    # TODO: 62로 바꿔서 다시하기.
                        help="Scene idx out of 62 to end.")
    args = parser.parse_args()

    print(f"starting 2nd preprocessing, start: {args.scene_idx_start} / end: {args.scene_idx_end}.")
    save_root = "data_preprocessed/SIT_processed_2nd"
    # save_root_dict = "data_preprocessed/SIT_processed_2nd/sit_v2_fps_2_5_frame_20_withRobot_dict"
    datas = glob("data_preprocessed/SIT_processed_1st/*.pt")     # fixed 241010
    datas.sort()
    max_num_agents = 0
    # text_encoder = TextEncoder_TinyBERT()
    scene_idx = -1
    # breakpoint()
    for scene_count, i in enumerate(range(len(datas))):     # 총 49개의 scene
        scene_idx += 1
        if scene_count < args.scene_idx_start or scene_count > args.scene_idx_end: continue
        agent_data = torch.load(datas[i])
        # breakpoint()
        if 'prompt' not in datas[i]:        # TODO: 여기 scene name 제대로 안들어가고 있음.
            scene_name = datas[i][36:-3]        # If you change the path, you have to change this too.
        else:
            scene_name = datas[i][36:-3]        # If you change the path, you have to change this too.
        print(f"Processing scene {scene_name}")
        # breakpoint()
        # if scene_name != "ocessed_1st/sit_v1_debug/Lobby_7-004_agents_0_to_200": 
        #     continue
        
        preprocess = PreProcess(agent_data, scene_name, scene_idx)
        frame_range = list(agent_data.keys())
        frame_range.sort()
        
        try:
            os.makedirs(f"{save_root}/{scene_name}")
            # os.makedirs(f"{save_root_dict}/{scene_name}")
        except:
            pass
        # breakpoint()
        # frame_range[-1] = 199
        # preprocess.FRAME_STEP = 4
        # (preprocess.FRAME_LENGTH - 1) = 19
        for initial_frame in tqdm(range(frame_range[-1] - preprocess.FRAME_STEP * (preprocess.FRAME_LENGTH - 1))):
            result, result_dict, num_agents = preprocess(initial_frame, None)
            if result is not None:
                torch.save(result, f"{save_root}/{scene_name}/{initial_frame}.pt")
                # torch.save(result_dict, f"{save_root_dict}/{scene_name}/{initial_frame}.pt")
                if num_agents > max_num_agents: max_num_agents = num_agents
        print(f"Max num agents: {max_num_agents}")
        # break # for temporal testing





