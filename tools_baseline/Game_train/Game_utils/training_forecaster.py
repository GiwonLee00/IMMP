# TODO: 250209에 옮김. 변경사항 생기면 다시 수정.
import torch
import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.viz_utils import *
from utils.metric import *

# NOTE: unimodal forecaster training
def train_forecaster_uni(forecaster, data_loader, criterion, optimizer,\
    history=8, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=None, device='cpu', viz_dir=None, epoch_number=None, args=None):
    if is_train:
        forecaster.train()
        loss_sum_all = 0
    else:
        forecaster.eval()
        ade_total, fde_total, agent_total, total_batch_size = 0, 0, 0, 0

    batch_idx = 0
    for batch_data in tqdm.tqdm(data_loader, total=len(data_loader)):
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person = batch_data["neg_seeds"], batch_data["neg_seeds_mask"], batch_data["neg_hist"], batch_data["neg_hist_mask"], batch_data["agent_mask"], batch_data["pad_mask_person"] 
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person = neg_seeds.cuda(), neg_seeds_mask.cuda(), neg_hist.cuda(), neg_hist_mask.cuda(), agent_mask.cuda(), pad_mask_person.cuda()
        # breakpoint()
        '''
        Data shape description
        neg_seeds: Batch, MaxAgent, Horizon, 2 (128,150,12,2)
        neg_seeds_mask: Batch, MaxAgent, Horizon, 1 (128,150,12,1)
        neg_hist: Batch, MaxAgent, History, 2 (128,150,8,2)
        neg_hist_mask: Batch, MaxAgent, History, 1 (128,150,8,1)
        agent_mask: Batch, MaxAgent (128,150)
        pad_mask_person: Batch, MaxAgent, MaxAgent (128,150,150)
        '''
        
        neg_seeds = neg_seeds*neg_seeds_mask
        neg_hist = neg_hist*neg_hist_mask
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,:,-1,:]  # B, N ,2

        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person, neg_hist_mask)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person, neg_hist_mask)

        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(outputs, dim = 2)    # B, N, F_T, 2

        # NOTE: vizualize
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(pred=forecasts*neg_seeds_mask, gt=neg_seeds*neg_seeds_mask, prev_gt=neg_hist*neg_hist_mask, output_dir=viz_dir, batch_idx=batch_idx, epoch_number=epoch_number)
        
        if is_train:
            loss = criterion(forecasts*neg_seeds_mask, neg_seeds*neg_seeds_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum_all += loss.data.item()
        else:
            ade, fde, valid_agent_number = get_forecast_errors(forecasts, neg_seeds, neg_seeds_mask)
            ade_total += torch.sum(ade)
            fde_total += torch.sum(fde)
            agent_total += torch.sum(valid_agent_number)  # 128,150중 얼마나 많은 valid agent가 있는지.
              
        batch_idx += 1
    if is_train:
        return loss_sum_all / batch_idx
    else:
        # return ade_total/total_batch_size, fde_total/total_batch_size
        return ade_total/agent_total, fde_total/agent_total


# NOTE: multimodal forecaster training
# TODO: 아직 수정 안했음.
def train_forecaster_multi(forecaster, train_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu', viz_dir = None):

    if is_train:
        forecaster.train()
    else:
        forecaster.eval()
    loss_sum_all = 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        #(robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, \
        #            pad_mask_person2d, pad_mask_person1d, robot_index, diff_h2h, diff_r2h, neg_hist_clone, pos_hist_clone, neg_seeds_clone_, pos_seeds_clone_) = batch
        # robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person, robot_idx = batch_data
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        # neg_mask = neg_mask.permute(0,2,1,3)
        # neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)  # NOTE: 내가 잠시 추가함.
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)
        human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        
        
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)
        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        
        forecasts = human_states[:, :, :2].unsqueeze(2).unsqueeze(2) + torch.cumsum(outputs, dim = 2)
        # breakpoint()
        min_loss = 999999999
        for i in range(forecasts.shape[2]):
            loss = criterion(forecasts[:,:,i,:,:]*neg_mask, neg_seeds*neg_mask) # uses vx, vy velocities
            if min_loss > loss: 
                min_loss = loss
                
        loss = min_loss
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if viz_dir is not None: viz_trajectory(forecasts*neg_mask, neg_seeds*neg_mask, neg_hist.permute(0,2,1,3)*neg_mask, viz_dir, batch_idx, agent_idx_=robot_idx)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_idx += 1
        loss_sum_all += loss.data.item()
    return loss_sum_all / batch_idx


def train_forecaster_validation_ETH_manyforecast(forecaster, valid_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=False, device = 'cpu', viz_dir = None, epoch_number = 0):

    forecaster.eval()
    loss_sum_all = 0
    batch_idx = 0
    ade_total, fde_total, total_batch_size = 0, 0, 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        # robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person, robot_idx = batch_data
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        # neg_mask = neg_mask.permute(0,2,1,3)
        # neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)
        human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)
        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2].unsqueeze(2).unsqueeze(2) + torch.cumsum(outputs, dim = 2)
        # breakpoint()
        # loss = criterion(forecasts*neg_mask, neg_seeds*neg_mask) # uses vx, vy velocities
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory_manyforecast(forecasts, neg_seeds, neg_hist.permute(0,2,1,3), viz_dir, batch_idx, agent_idx_=robot_idx, epoch_number = epoch_number)
        # if is_train:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        batch_idx += 1

        mean_ade = 0
        mean_fde = 0
        for i in range(outputs.shape[1]):
            for j in range(outputs.shape[2]):
                ade, fde = get_forecast_errors(forecasts[:,i,j].unsqueeze(1), neg_seeds[:,i].unsqueeze(1))
                mean_ade += ade
                mean_fde += fde
        ade_total += mean_ade / (outputs.shape[1] * outputs.shape[2])
        fde_total += mean_fde / (outputs.shape[1] * outputs.shape[2])
        total_batch_size += robot_states.shape[0]
    
    return ade_total/total_batch_size, fde_total/total_batch_size

