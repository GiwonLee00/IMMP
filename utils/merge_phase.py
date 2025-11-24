import torch
import tqdm
from utils.viz_utils import *
from utils.metric import *   
import sys
sys.path.append("..")
from policy.planner_model_uni_merge import PlannerModel_uni_merge


def merge_phase(is_train=None, cfg=None, args=None, train_loader=None, device=None, viz_dir=None, epoch_number=None, col_weight=None, merged_model=None, optimizer=None, scheduler=None, criterion=None):
    if is_train:
        merged_model.train()
        loss_sum_all, loss_sum_reg, loss_sum_col = 0, 0, 0
    else:
        merged_model.eval()
        ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    

    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = batch_data["neg_seeds"], batch_data["neg_seeds_mask"], batch_data["neg_hist"], batch_data["neg_hist_mask"], batch_data["agent_mask"], batch_data["pad_mask_person"], batch_data["pos_seeds"], batch_data["pos_seeds_mask"], batch_data["pos_hist"], batch_data["pos_hist_mask"] 
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = neg_seeds.cuda(), neg_seeds_mask.cuda(), neg_hist.cuda(), neg_hist_mask.cuda(), agent_mask.cuda(), pad_mask_person.cuda(), pos_seeds.cuda(), pos_seeds_mask.cuda(), pos_hist.cuda(), pos_hist_mask.cuda()  

        human_history = neg_hist   # B, N, H_T, 2
        human_states = neg_hist[:,:,-1,:]  # B, N, 2
        robot_states = pos_hist[:,-1,:]  # B, 2
        robot_states = torch.cat([robot_states, pos_hist[:, -1] - pos_hist[:, -2]], dim = -1) # B, 4
        
        if is_train:
            vel_plan = merged_model(human_states, robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask)     
        else:
            with torch.no_grad():
                vel_plan = merged_model(human_states, robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask)

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist.unsqueeze(1), viz_dir, batch_idx, agent_idx_=0, epoch_number=epoch_number)
        
        if is_train:
            reg_loss = criterion(plan*pos_seeds_mask, pos_seeds.squeeze(1)) 
            col_loss = col_weight*get_chomp_col_loss_ETH(plan, neg_seeds, neg_seeds_mask)
            loss = reg_loss + col_loss
            
            loss_sum_all += loss.data.item()
            loss_sum_reg += reg_loss.data.item()
            loss_sum_col += col_loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss_sum_all)

            # NOTE: parmeter가 적절히 학습되는지 여부 확인용 프린트문
            # print(merged_model.collect_trainable_params(), end="", flush=True)

        else:
            ade, fde, col, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_seeds_mask, plans_sampled=plan.unsqueeze(1), plans_gt=pos_seeds, human_y_gt=neg_seeds, plan_horizon=pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
            datasetsize = robot_states.shape[0]
            ade_total += sum(ade)
            fde_total += sum(fde)
            col_total += sum(col)
            jerk_total += sum(jerk)
            miss_rate_total += sum(miss_rate)
            jerkrate_total += sum(jerkrate)
            total_datset_size += datasetsize
        
        batch_idx += 1

    if is_train:
        scheduler.step(loss_sum_all)
        return loss_sum_all / batch_idx, loss_sum_reg / batch_idx, loss_sum_col / batch_idx, merged_model
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

    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

