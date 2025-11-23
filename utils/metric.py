import torch
# NOTE: metric code

def get_forecast_errors(forecasts, neg_seeds, neg_seeds_mask):
    '''
    data shape guideline
    forecasts:  128, 150, 12, 2   
    neg_seeds:  128, 150, 12, 2
    neg_seeds_mask: 128, 150, 12, 1
    '''
    # breakpoint()
    PRED = forecasts*neg_seeds_mask
    GT = neg_seeds*neg_seeds_mask
    errors = torch.linalg.norm(PRED-GT, dim = -1)       # 128, 150, 12
    valid_agent_mask = (neg_seeds_mask.sum(dim=(-2, -1)) > 0)
    ade = (torch.sum(errors, dim=-1) / torch.sum(neg_seeds_mask.squeeze(-1), dim=-1))[valid_agent_mask]      # 각 agent 마다의 ade를 구해줌
    fde = errors[:, :, -1]          # 마지막 agent가 없는 경우, fde = 0으로 계산.
    return ade, fde, valid_agent_mask

def get_return_total(neg_mask, plans_sampled, plans_gt, human_y_gt, plan_horizon, human_y_pred=None, col_threshold = None, miss_threshold = None, train_mode = False, robot_pos=None, human_pos=None):
    '''
    data shape guideline
    neg_mask: B, N, F_T, 1      # SIT, JRDB
    neg_mask: B, N, F_T, 2      # ETH
    plans_sampled: B, 1, F_T, 2
    plans_gt: B, 1, F_T, 2
    human_y_gt: B, N, F_T, 2
    '''
    # breakpoint()
    col_threshold=0.6                 # crowdnav: 0.6 ETH: 0.2, TODO: (SIT: grid 마다의 실제 거리로 넣어주자.) # 0.6으로 통일해서 넣어주자
    neg_mask_ = neg_mask.sum(dim=-1) > 0        # B, N, F_T

    # TODO: ETH, crowdnav에서는 이거랑 또 다를 것.... 한번 확인해봐.
    # neg_mask_ = neg_mask.reshape(neg_mask.shape[0], neg_mask.shape[1], -1)      
    # neg_mask_ = neg_mask_.sum(-1) == plan_horizon*2
    # neg_mask = neg_mask.sum(-1) == 2
    # if len(plans_sampled.shape) == 3: plans_sampled = plans_sampled.unsqueeze(1)
    
    # collision with gt
    distances = torch.linalg.norm(plans_sampled-human_y_gt, dim=-1)
    is_col = distances < col_threshold      # B, N, F_T
    is_col_gt = is_col > 0 
    is_col_gt[~neg_mask_] = 0           # SIT, JRDB have invalid agent in timestep. masking it as zero
    is_col_gt = is_col_gt.sum(dim=(1, 2))  
    is_col_gt = is_col_gt / neg_mask_.sum(dim=(1, 2))  # NOTE: SCENE 마다 valid한 point의 수만큼 나눠서 비율로 구함.
    
    ## ADEs
    errors = torch.linalg.norm(plans_sampled-plans_gt, dim = -1).squeeze(1)
    ade = torch.mean(errors, dim=-1)
    fde = errors[...,-1]

    ## Miss rate
    miss_threshold=0.5              # 평가기준입니다. validation에서는 이 값이 쓰이고, 학습 시 따로 지정 안해주면 이 값으로 쓰임.
    distance_to_ref = errors[:,-1]
    miss_rate = (distance_to_ref > miss_threshold)
    
    # Jerk
    timestep = 0.4  # fps 맞춰주자!!! 
    v_x, v_y = torch.diff(plans_sampled.squeeze(1)[:, :, 0], dim=1) / timestep , torch.diff(plans_sampled.squeeze(1)[:, :, 1], dim=1) / timestep
    a_x, a_y = torch.diff(v_x, dim=1) / timestep , torch.diff(v_y, dim=1) / timestep
    j_x, j_y = torch.diff(a_x, dim=1) / timestep , torch.diff(a_y, dim=1) / timestep

    jerk = torch.sqrt(torch.square(j_x) + torch.square(j_y))
    jerk = torch.mean(torch.abs(jerk), axis = -1)
    jerk_threshold = 2.0
    jerkrate = jerk > jerk_threshold
    # breakpoint()
    return ade, fde, is_col_gt, jerk, miss_rate, jerkrate