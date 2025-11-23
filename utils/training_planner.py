import torch
import tqdm
from utils.viz_utils import *
from utils.metric import *   
import sys
sys.path.append("..")
from policy.planner_model_uni_merge import PlannerModel_uni_merge

def train_planner_uni_merge(is_train=None, cfg=None, args=None, train_loader=None, device=None, viz_dir=None, epoch_number=None, col_weight=None, MergingModel=None):
    if is_train:
        MergingModel.train()
        loss_sum_all, loss_sum_reg, loss_sum_col = 0, 0, 0
    else:
        MergingModel.eval()
        ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    

    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        if is_train:
            loss, lossreg, losscol = MergingModel(batch_data=batch_data, batch_idx=batch_idx, viz_dir=viz_dir, epoch_number=epoch_number, col_weight=col_weight, is_train=True)
            loss_sum_all += loss
            loss_sum_reg += lossreg
            loss_sum_col += losscol
        else:
            # breakpoint()
            ade, fde, col, jerk, missrate, jerkrate, datasetsize = MergingModel(batch_data=batch_data, batch_idx=batch_idx, viz_dir=viz_dir, epoch_number=epoch_number, col_weight=col_weight, is_train=False)
            ade_total += ade
            fde_total += fde
            col_total += col
            jerk_total += jerk
            miss_rate_total += missrate
            jerkrate_total += jerkrate
            total_datset_size += datasetsize
        batch_idx += 1

    if is_train:
        MergingModel.planner_scheduler.step(loss_sum_all)
    if is_train:
        return loss_sum_all / batch_idx, loss_sum_reg / batch_idx, loss_sum_col / batch_idx, MergingModel
    else:
        return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size, jerkrate_total/total_datset_size


def train_planner_uni(forecaster, planner, train_loader, criterion, optimizer,\
         num_human=5, history=8, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device='cpu', viz_dir=None, epoch_number=0, args=None):
    forecaster.eval()
    if is_train:
        planner.train()
        loss_sum_all, loss_sum_reg, loss_sum_col = 0, 0, 0
    else:
        planner.eval()
        ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    
    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = batch_data["neg_seeds"], batch_data["neg_seeds_mask"], batch_data["neg_hist"], batch_data["neg_hist_mask"], batch_data["agent_mask"], batch_data["pad_mask_person"], batch_data["pos_seeds"], batch_data["pos_seeds_mask"], batch_data["pos_hist"], batch_data["pos_hist_mask"] 
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = neg_seeds.cuda(), neg_seeds_mask.cuda(), neg_hist.cuda(), neg_hist_mask.cuda(), agent_mask.cuda(), pad_mask_person.cuda(), pos_seeds.cuda(), pos_seeds_mask.cuda(), pos_hist.cuda(), pos_hist_mask.cuda()  
        # breakpoint() # pos_seeds_mask: 128, 12, 1
        '''
        Data description
        neg_seeds: Batch, MaxAgent, Horizon, 2 (128,150,12,2)
        neg_seeds_mask: Batch, MaxAgent, Horizon, 1 (128,150,12,1)
        neg_hist: Batch, MaxAgent, History, 2 (128,150,8,2)
        neg_hist_mask: Batch, MaxAgent, History, 1 (128,150,8,1)
        agent_mask: Batch, MaxAgent (128,150)
        pad_mask_person: Batch, MaxAgent, MaxAgent (128,150,150)
        pos_seeds: 128, 1, 12, 2
        pos_seeds_mask: 128, 12 , 1
        pos_hist: 128, 8, 2
        pos_hist_mask: 128, 8, 1
        '''

        human_history = neg_hist   # B, N, H_T, 2
        human_states = neg_hist[:,:,-1,:]  # B, N, 2
        robot_states = pos_hist[:,-1,:]  # B, 2
        robot_states = torch.cat([robot_states, pos_hist[:, -1] - pos_hist[:, -2]], dim = -1) # B, 4
        
        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person, neg_hist_mask)
            forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
     
        if is_train:
            vel_plan = planner(robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=forecasts)     
        else:
            with torch.no_grad():
                vel_plan = planner(robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=forecasts)  # TODO: 평가에서는 mask 사용 불가?.

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist.unsqueeze(1), viz_dir, batch_idx, agent_idx_=0, epoch_number=epoch_number)
        
        if is_train:
            reg_loss = criterion(plan*pos_seeds_mask, pos_seeds.squeeze(1))              # pos seeds와 비교해서 loss를 줌
            col_loss = col_weight*get_chomp_col_loss_ETH(plan, neg_seeds, neg_seeds_mask)
            loss = reg_loss + col_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum_all += loss.data.item()
            loss_sum_reg += reg_loss.item()
            loss_sum_col += col_loss.item()
        else:
            ade, fde, col_gt, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_seeds_mask, plans_sampled=plan.unsqueeze(1), plans_gt=pos_seeds, human_y_gt=neg_seeds, plan_horizon=pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
            ade_total += sum(ade)
            fde_total += sum(fde)
            col_total += sum(col_gt)
            jerk_total += sum(jerk)
            miss_rate_total += sum(miss_rate)
            jerkrate_total += sum(jerkrate)
            total_datset_size += robot_states.shape[0]
        batch_idx += 1

    if is_train:
        return loss_sum_all / batch_idx, loss_sum_reg / batch_idx, loss_sum_col / batch_idx
    else:
        return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size, jerkrate_total/total_datset_size

def get_chomp_col_loss_ETH(plan, forecasts, neg_mask, threshold = 0.6, eps = 0.2):
    maskkk = neg_mask.clone().float()           # [3][6]        # float형태로 들어와야 함.
    maskkk[maskkk == 0] = 999999999
    distances = torch.linalg.norm((plan.unsqueeze(1)-forecasts)*maskkk, dim=-1)
    
    l1_col_mask = distances < threshold     # 128,150,12
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))       

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)
    # breakpoint()
    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

