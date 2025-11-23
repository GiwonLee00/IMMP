import torch
import policy.planner_diffusion 
import policy.planner_encoder 
import tqdm
from utils.viz_utils import *
from utils.metric import *

def train_planner(planner, train_loader, criterion, optimizer,\
         num_human=5, horizon=12, col_weight = 0.1, col_threshold = 0.6):
    planner.train()
    
    loss_sum_all, loss_sum_task, loss_sum_col = 0, 0, 0

    for robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist in train_loader:
        
        plan = planner(robot_states, human_states)

        plan_vel = plan[:, 1:, :] - plan[:, :-1, :]
        pos_vel = pos_seeds[:, 1:, :] - pos_seeds[:, :-1, :]
        
        plan_first_vel = plan[:, 0, :] - robot_states[:, :2]
        a, c = plan_first_vel.shape
        plan_first_vel = plan_first_vel.view((a, 1, c))
        pos_first_vel = pos_seeds[:, 0, :] - robot_states[:, :2]
        pos_first_vel = pos_first_vel.view((a, 1, c))
        
        plan_vel = torch.cat((plan_first_vel, play_vel), 1)
        pos_vel = torch.cat((pos_first_vel, pos_vel), 1)

        # loss_task = criterion(plan, pos_seeds) # uses x,y positions
        loss_task = criterion(plan_vel, pos_vel) # uses vx, vy velocities
        
        col_loss = col_weight*get_chomp_col_loss(plan, neg_seeds.permute([0, 2, 1, 3]))
        loss_sum_col += col_loss.item()

        loss = loss_task + col_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum_all += loss.data.item()
        loss_sum_task += loss_task.item()
        loss_sum_col += col_loss.item()

    num_batch = len(train_loader)

    return loss_sum_all / num_batch, loss_sum_task / num_batch, \
        loss_sum_col / num_batch

def train_forecaster(forecaster, train_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu'):

    if is_train:
        forecaster.train()
    else:
        forecaster.eval()
    loss_sum_all = 0
    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch
        
        human_states = human_states.to(device)
        neg_seeds = neg_seeds.to(device)
        neg_hist = neg_hist.to(device) 

        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)
        if is_train:
            outputs, _ = forecaster(human_history)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(outputs, dim = 2)  

        neg_seeds = neg_seeds.permute([0, 2, 1, 3])
        loss = criterion(forecasts, neg_seeds) # uses vx, vy velocities

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch

def train_forecaster_validation(forecaster, valid_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu'):
    forecaster.eval()
    loss_sum_all = 0
    ade_total, fde_total, total_batch_size = 0, 0, 0
    for batch in valid_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch
        
        human_states = human_states.to(device)
        neg_seeds = neg_seeds.to(device)
        neg_hist = neg_hist.to(device) 
        
        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)
        if is_train:
            outputs, _ = forecaster(human_history)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(outputs, dim = 2)  

        neg_seeds = neg_seeds.permute([0, 2, 1, 3])
        loss = criterion(forecasts, neg_seeds) # uses vx, vy velocities
        #  torch.nonzero(neg_mask == 0, as_tuple=False)
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()
        ade, fde = get_forecast_errors(forecasts, neg_seeds)
        ade_total += ade
        fde_total += fde
        total_batch_size += robot_states.shape[0]
    
    return ade_total/total_batch_size, fde_total/total_batch_size

def train_forecaster_ETH(forecaster, train_loader, criterion, optimizer,\
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
         # NOTE: 여기 내가 새로운 데이터 배치에 맞게 일부 수정하였음.
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        neg_mask = neg_mask.permute(0,2,1,3)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)  # robotstates는 사용 x
        neg_hist = neg_hist.permute(0,2,1,3)
        # human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        
        
        # human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)    # 128, 9, 60, 2
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(outputs, dim = 2)

        loss = criterion(forecasts.permute(0,2,1,3)*neg_mask, neg_seeds*neg_mask) # uses vx, vy velocities
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if viz_dir is not None: viz_trajectory(forecasts.permute(0,2,1,3)*neg_mask, neg_seeds*neg_mask, neg_hist.permute(0,2,1,3)*neg_mask, viz_dir, batch_idx, agent_idx_=robot_idx)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_idx += 1
        loss_sum_all += loss.data.item()
    return loss_sum_all / batch_idx

def train_forecaster_validation_ETH(forecaster, valid_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=False, device = 'cpu', viz_dir = None, epoch_number = 0):

    forecaster.eval()
    loss_sum_all = 0
    batch_idx = 0
    ade_total, fde_total, total_batch_size = 0, 0, 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        # robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person, robot_idx = batch_data
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        # neg_mask = neg_mask.permute(0,2,1,3)
        # neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)  # NOTE: 내가 잠시 추가함.
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)
        # human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)

        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        # human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(outputs, dim = 2)
        # breakpoint()
        # loss = criterion(forecasts*neg_mask, neg_seeds*neg_mask) # uses vx, vy velocities
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(forecasts, neg_seeds, neg_hist.permute(0,2,1,3), viz_dir, batch_idx, agent_idx_=robot_idx, epoch_number = epoch_number)
        # if is_train:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        batch_idx += 1


        # breakpoint()
        for batch_number in range(outputs.shape[0]):
            mean_ade = 0
            mean_fde = 0
            for i in range(outputs.shape[1]):   # forecasts:
                ade, fde = get_forecast_errors((forecasts[batch_number,i]*neg_mask[batch_number,i]).unsqueeze(0).unsqueeze(0), (neg_seeds[batch_number,i]*neg_mask[batch_number,i]).unsqueeze(0).unsqueeze(0))
                mean_ade += ade
                mean_fde += fde
            realhuman_number = np.sum(neg_mask[batch_number].cpu().numpy() == 1) / (neg_mask.shape[-1] * neg_mask.shape[-2]) 
            
            # 유효한 한 scene의 사람 수 구하고
            # 이를 mean ade 값으로 나눠서 scene마다의 평균 ade를 구함. 
            ade_total += mean_ade/realhuman_number
            fde_total += mean_fde/realhuman_number
            # 지금은 scene마다의 ade가 업데이트 되고 있는 것이다. 
        total_batch_size += robot_states.shape[0]
    
    return ade_total/total_batch_size, fde_total/total_batch_size


def train_forecaster_SIT(forecaster, train_loader, criterion, optimizer,\
    num_human=5, history=8, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu', viz_dir = None):

    if is_train:
        forecaster.train()
    else:
        forecaster.eval()
    loss_sum_all = 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = batch_data 
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = padded_agent.cuda(), agent_path_mask.cuda(), agent_mask.cuda(), agent_2D_mask.cuda(), robot_pos.cuda(), robot_mask.cuda()
        
        neg_seeds = padded_agent[:,:,history:,:]         # 32, 50, 12, 2 
        neg_seeds_mask = agent_path_mask[:,:,history:]   # 32, 50, 12
        neg_hist = padded_agent[:,:,:history,:]          # 32, 50, 8, 2
        neg_hist_mask = agent_path_mask[:,:,:history]    # 32, 50, 8
        pad_mask_person = agent_2D_mask

        neg_seeds_mask = neg_seeds_mask.permute(0,2,1).unsqueeze(-1)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        neg_hist_mask = neg_hist_mask.permute(0,2,1)

        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        # breakpoint()
        # breakpoint()
        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(outputs, dim = 2)    # 32, 50, 12, 2
        # breakpoint()
        loss = criterion(forecasts.permute(0,2,1,3)*neg_seeds_mask, neg_seeds*neg_seeds_mask) # uses vx, vy velocities
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if viz_dir is not None: viz_trajectory(forecasts.permute(0,2,1,3)*neg_seeds_mask, neg_seeds*neg_seeds_mask, neg_hist.permute(0,2,1,3)*neg_seeds_mask, viz_dir, batch_idx, agent_idx_=robot_idx)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_idx += 1
        loss_sum_all += loss.data.item()
    return loss_sum_all / batch_idx

def train_forecaster_validation_SIT(forecaster, valid_loader, criterion, optimizer,\
    num_human=5, history=8, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=False, device = 'cpu', viz_dir = None, epoch_number = 0):

    forecaster.eval()
    loss_sum_all = 0
    batch_idx = 0
    ade_total, fde_total, total_batch_size = 0, 0, 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = batch_data 
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = padded_agent.cuda(), agent_path_mask.cuda(), agent_mask.cuda(), agent_2D_mask.cuda(), robot_pos.cuda(), robot_mask.cuda()
        
        neg_seeds = padded_agent[:,:,history:,:]         # 32, 50, 12, 2 
        neg_seeds_mask = agent_path_mask[:,:,history:]   # 32, 50, 12
        neg_hist = padded_agent[:,:,:history,:]          # 32, 50, 8, 2
        neg_hist_mask = agent_path_mask[:,:,:history]    # 32, 50, 8
        pad_mask_person = agent_2D_mask

        neg_seeds_mask = neg_seeds_mask.unsqueeze(-1)
        # neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        neg_hist_mask = neg_hist_mask.permute(0,2,1)

        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2

        if is_train:
            outputs, _ = forecaster(human_history, pad_mask_person)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(outputs, dim = 2)
        # breakpoint()
        # loss = criterion(forecasts*neg_mask, neg_seeds*neg_mask) # uses vx, vy velocities
        # neg_seeds.shape -> 1,1,12,2      두번째가 아마 scene에서 human의 수. 
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(forecasts, neg_seeds, neg_hist.permute(0,2,1,3), viz_dir, batch_idx, agent_idx_=0, epoch_number = epoch_number)
        # if is_train:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        batch_idx += 1


        # breakpoint()
        # neg_seeds_mask = neg_seeds_mask.permute(0, 2, 1, 3)
        for batch_number in range(outputs.shape[0]):
            mean_ade = 0
            mean_fde = 0
            for i in range(outputs.shape[1]):   # forecasts:
                ade, fde = get_forecast_errors((forecasts[batch_number,i]*neg_seeds_mask[batch_number,i]).unsqueeze(0).unsqueeze(0), (neg_seeds[batch_number,i]*neg_seeds_mask[batch_number,i]).unsqueeze(0).unsqueeze(0))
                mean_ade += ade
                mean_fde += fde
            # breakpoint()
            realhuman_number = np.sum(agent_mask[batch_number].cpu().numpy() == 1) 
            
            # 유효한 한 scene의 사람 수 구하고
            # 이를 mean ade 값으로 나눠서 scene마다의 평균 ade를 구함. 
            ade_total += mean_ade/realhuman_number
            fde_total += mean_fde/realhuman_number
            # 지금은 scene마다의 ade가 업데이트 되고 있는 것이다. 
        # breakpoint()
        total_batch_size += outputs.shape[0]
    
    return ade_total/total_batch_size, fde_total/total_batch_size

def train_forecaster_ETH_manyforecasts(forecaster, train_loader, criterion, optimizer,\
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



def train_forecaster_fixed(forecaster, train_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu'):

    if is_train:
        forecaster.train()
    else:
        forecaster.eval()
    loss_sum_all = 0
    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch
        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)
        if is_train:
            # outputs, _ = forecaster(human_history)
            # num_path 항을 추가했습니다. (여러개의 forecast를 생성하도록 함.)
            outputs, _ = forecaster(human_history)
        else:
            with torch.no_grad():
                # outputs, _ = forecaster(human_history)
                outputs, _ = forecaster(human_history)
        # forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(outputs, dim = 2)    # dim이 아마 3으로 바뀌어야 할 듯??? (원래는 2)
        forecasts = human_states[:, :, :2].unsqueeze(2).unsqueeze(2) + torch.cumsum(outputs, dim = 3)    # human states도 바꿔줘야 할지도???? 저게 sum이 안될 거 같음. 체크포인트
        # 128, 5, 6, 8, 2

        neg_seeds = neg_seeds.permute([0, 2, 1, 3])
        loss = 0
        path_list = torch.unbind(forecasts, dim=2)    # [128, 5, 8, 2] 짜리가 num_path 개수 만큼 리스트에 담김
        semi_loss = 1000000
        for forecast_tensor in path_list:
            # loss += criterion(forecast_tensor*mask, neg_seeds*mask) # uses vx, vy velocities
            exist_loss = criterion(forecast_tensor*mask, neg_seeds*mask) # uses vx, vy velocities
            if semi_loss > exist_loss:
                semi_loss = exist_loss
                # print("current exist loss: ", exist_loss)
                # print("current semi loss: ", semi_loss)

        loss = semi_loss

        if is_train: 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch

def train_planner_forecaster_disjointly(forecaster, planner, train_loader, criterion, optimizer,\
         num_human=5, horizon=12, col_weight = 0.0, col_threshold = 0.6, is_train=True, device = 'cpu'):

    forecaster.eval()
    if is_train:
        planner.train()
    else:
        planner.eval()
    
    loss_sum_all, loss_sum_task, loss_sum_col = 0, 0, 0

    for batch in train_loader:  # 여기!
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch
        robot_states = robot_states.to(device)
        human_states = human_states.to(device)
        pos_seeds = pos_seeds.to(device)
        neg_hist = neg_hist.to(device)

        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        with torch.no_grad():
            vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)

        if is_train:
            vel_plan = planner(robot_states, human_history, forecasts)     
        else:
            with torch.no_grad():
                vel_plan = planner(robot_states, human_history, forecasts)

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성


        loss = criterion(plan, pos_seeds)              # pos seeds와 비교해서 loss를 줌
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()

    num_batch = len(train_loader)

    return loss_sum_all / num_batch

def train_planner_forecaster_disjointly_validation(forecaster, planner, valid_loader, horizon=8, device = 'cpu'):
    forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_total, cost_total, total_datset_size = 0, 0, 0, 0, 0
    for batch in valid_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch
        del batch
        robot_states = robot_states.to(device)
        human_states = human_states.to(device)
        pos_seeds = pos_seeds.to(device)
        neg_seeds = neg_seeds.to(device)
        neg_hist = neg_hist.to(device)
        
        neg_hist = neg_hist.permute([0, 2, 1, 3])    
        neg_seeds = neg_seeds.permute([0, 2, 1, 3])    
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        vel_forecasts, _ = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)
        ade, fde, col, cost = get_return(plan, pos_seeds, forecasts, horizon)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_total += sum(col)
        cost_total += sum(cost)
        total_datset_size += robot_states.shape[0]
        del plan
        del vel_plan
        del forecasts
        del vel_forecasts
        del human_history
        del neg_seeds
        del pos_seeds
        del human_states
        del robot_states
        del action_targets
        del pos_hist
        del neg_hist
        torch.cuda.empty_cache()
        
    # breakpoint()
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, cost_total/total_datset_size


def train_planner_forecaster_disjointly_ETH(forecaster, planner, train_loader, criterion, optimizer,\
         num_human=5, horizon=12, col_weight = 0.0, col_threshold = 0.6, is_train=True, device = 'cpu'):

    forecaster.eval()
    if is_train:
        planner.train()
    else:
        planner.eval()
    
    loss_sum_all = 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        # neg_mask = neg_mask.permute(0,2,1,3)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)  # robotstates는 사용 x
        neg_hist = neg_hist.permute(0,2,1,3)
        # human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        # human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)    # 128, 9, 60, 2
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,60,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 
        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)

        if is_train:
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)     
        else:
            with torch.no_grad():
                vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성

        # breakpoint()
        loss = criterion(plan, pos_seeds.squeeze(1))              # pos seeds와 비교해서 loss를 줌
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_idx += 1
        loss_sum_all += loss.data.item()

    return loss_sum_all / batch_idx

def train_planner_forecaster_disjointly_validation_ETH(forecaster, planner, valid_loader, horizon=8, device = 'cpu', viz_dir = None, epoch_number = 0):
    forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        # neg_mask = neg_mask.permute(0,2,1,3)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        # robot_states = torch.cat((robot_states, pos_hist[:, -1] - pos_hist[:, -2]), dim=-1)  # robotstates는 사용 x
        neg_hist = neg_hist.permute(0,2,1,3)
        # human_states = torch.cat((human_states, neg_hist[:, -1] - neg_hist[:, -2]), dim=-1)
        # robot_states, human_states, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask = \
        #     robot_states, human_states.unsqueeze(0), pos_seeds, neg_seeds.unsqueeze(0), pos_hist.unsqueeze(0), neg_hist.unsqueeze(0), neg_mask.unsqueeze(0) 
        # human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(1)], dim = 1)    # 128, 9, 60, 2
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 

        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person)
            forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)   

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        # breakpoint()
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist, viz_dir, batch_idx, agent_idx_=robot_idx, epoch_number=epoch_number)
        batch_idx += 1
        # ade, fde, col, cost = get_return_total(plan, pos_seeds.squeeze(1), neg_seeds, horizon)  # 일단 negseeds로 평가하도록 하자.
        # self, neg_mask, plans_sampled, plans_gt, human_y_gt, plan_horizon, 
        ade, fde, col_gt, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_mask, plans_sampled = plan, plans_gt = pos_seeds.squeeze(1), human_y_gt = neg_seeds, plan_horizon = pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_total += sum(col_gt)
        jerk_total += sum(jerk)
        miss_rate_total += sum(miss_rate)
        jerkrate_total += sum(jerkrate)
        total_datset_size += robot_states.shape[0]
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size, jerkrate_total/total_datset_size



def train_planner_forecaster_disjointly_SIT(forecaster, planner, train_loader, criterion, optimizer,\
         num_human=5, history=8, horizon=12, col_weight = 0.0, col_threshold = 0.6, is_train=True, device = 'cpu'):

    forecaster.eval()
    if is_train:
        planner.train()
    else:
        planner.eval()
    
    loss_sum_all = 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = batch_data 
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = padded_agent.cuda(), agent_path_mask.cuda(), agent_mask.cuda(), agent_2D_mask.cuda(), robot_pos.cuda(), robot_mask.cuda()
        
        neg_seeds = padded_agent[:,:,history:,:]         # 32, 50, 12, 2 
        neg_seeds_mask = agent_path_mask[:,:,history:]   # 32, 50, 12
        neg_hist = padded_agent[:,:,:history,:]          # 32, 50, 8, 2
        neg_hist_mask = agent_path_mask[:,:,:history]    # 32, 50, 8
        pad_mask_person = agent_2D_mask
        # breakpoint()
        pos_hist = robot_pos[:,:history,:]
        pos_hist_mask = robot_mask[:,:history]
        pos_seeds = robot_pos[:,history:,:] 
        pos_seeds_mask = robot_mask[:,history:]

        neg_seeds_mask = neg_seeds_mask.unsqueeze(-1)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        neg_hist_mask = neg_hist_mask.permute(0,2,1)
        pos_seeds_mask = pos_seeds_mask.unsqueeze(-1)
                
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,60,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 
        
        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
     
        if is_train:
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_seeds_mask)     
        else:
            with torch.no_grad():
                vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_seeds_mask)

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        # breakpoint()
        loss = criterion(plan*pos_seeds_mask, pos_seeds.squeeze(1))              # pos seeds와 비교해서 loss를 줌
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_idx += 1
        loss_sum_all += loss.data.item()

    return loss_sum_all / batch_idx

def train_planner_forecaster_disjointly_validation_SIT(forecaster, planner, valid_loader, history=8, horizon=12, device = 'cpu', viz_dir = None, epoch_number = 0):
    forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = batch_data 
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = padded_agent.cuda(), agent_path_mask.cuda(), agent_mask.cuda(), agent_2D_mask.cuda(), robot_pos.cuda(), robot_mask.cuda()
        
        neg_seeds = padded_agent[:,:,history:,:]         # 32, 50, 12, 2 
        neg_seeds_mask = agent_path_mask[:,:,history:]   # 32, 50, 12
        neg_hist = padded_agent[:,:,:history,:]          # 32, 50, 8, 2
        neg_hist_mask = agent_path_mask[:,:,:history]    # 32, 50, 8
        pad_mask_person = agent_2D_mask
        # breakpoint()
        pos_hist = robot_pos[:,:history,:].unsqueeze(1)
        pos_hist_mask = robot_mask[:,:history]
        pos_seeds = robot_pos[:,history:,:].unsqueeze(1)
        pos_seeds_mask = robot_mask[:,history:]

        neg_seeds_mask = neg_seeds_mask.unsqueeze(-1)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        neg_hist_mask = neg_hist_mask.permute(0,2,1)
        pos_seeds_mask = pos_seeds_mask.unsqueeze(-1)
                
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 

        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person)
            forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_seeds_mask)   

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        # breakpoint()
        
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist, viz_dir, batch_idx, agent_idx_=0, epoch_number=epoch_number)
        batch_idx += 1
        # ade, fde, col, cost = get_return_total(plan, pos_seeds.squeeze(1), neg_seeds, horizon)  # 일단 negseeds로 평가하도록 하자.
        # self, neg_mask, plans_sampled, plans_gt, human_y_gt, plan_horizon, 
        ade, fde, col_gt, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_seeds_mask, plans_sampled = plan, plans_gt = pos_seeds.squeeze(1), human_y_gt = neg_seeds, plan_horizon = pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_total += sum(col_gt)
        jerk_total += sum(jerk)
        miss_rate_total += sum(miss_rate)
        jerkrate_total += sum(jerkrate)
        total_datset_size += robot_states.shape[0]
    # breakpoint()
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size, jerkrate_total/total_datset_size

def train_planner_forecaster_disjointly_validation_JRDB(forecaster, planner, valid_loader, history=8, horizon=12, device = 'cpu', viz_dir = None, epoch_number = 0):
    forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_total, jerk_total, miss_rate_total, jerkrate_total, total_datset_size = 0, 0, 0, 0, 0, 0, 0
    batch_idx = 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = batch_data 
        padded_agent, agent_path_mask, agent_mask, agent_2D_mask, robot_pos, robot_mask = padded_agent.cuda(), agent_path_mask.cuda(), agent_mask.cuda(), agent_2D_mask.cuda(), robot_pos.cuda(), robot_mask.cuda()
        
        neg_seeds = padded_agent[:,:,history:,:]         # 32, 50, 12, 2 
        neg_seeds_mask = agent_path_mask[:,:,history:]   # 32, 50, 12
        neg_hist = padded_agent[:,:,:history,:]          # 32, 50, 8, 2
        neg_hist_mask = agent_path_mask[:,:,:history]    # 32, 50, 8
        pad_mask_person = agent_2D_mask
        # breakpoint()
        pos_hist = robot_pos[:,:history,:].unsqueeze(1)
        pos_hist_mask = robot_mask[:,:history]
        pos_seeds = robot_pos[:,history:,:].unsqueeze(1)
        pos_seeds_mask = robot_mask[:,history:]

        neg_seeds_mask = neg_seeds_mask.unsqueeze(-1)
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        neg_hist_mask = neg_hist_mask.permute(0,2,1)
        pos_seeds_mask = pos_seeds_mask.unsqueeze(-1)
        # breakpoint()
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        # breakpoint()

        robot_states = pos_hist.squeeze(1)[:,-1,:]  # 128,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze(1)[:, -1] - pos_hist.squeeze(1)[:, -2]], dim = -1) 
        # breakpoint()

        # breakpoint()
        with torch.no_grad():
            vel_forecasts, _ = forecaster(human_history, pad_mask_person)
            forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_seeds_mask)   

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        # breakpoint()
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist, viz_dir, batch_idx, agent_idx_=0, epoch_number=epoch_number)
        batch_idx += 1
        # ade, fde, col, cost = get_return_total(plan, pos_seeds.squeeze(1), neg_seeds, horizon)  # 일단 negseeds로 평가하도록 하자.
        # self, neg_mask, plans_sampled, plans_gt, human_y_gt, plan_horizon, 
        ade, fde, col_gt, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_seeds_mask, plans_sampled = plan, plans_gt = pos_seeds.squeeze(1), human_y_gt = neg_seeds, plan_horizon = pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_total += sum(col_gt)
        jerk_total += sum(jerk)
        miss_rate_total += sum(miss_rate)
        jerkrate_total += sum(jerkrate)
        total_datset_size += robot_states.shape[0]
    # breakpoint()
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size, jerkrate_total/total_datset_size


def get_chomp_col_loss(plan, forecasts, threshold = 0.6, eps = 0.2):
    distances = torch.linalg.norm(plan.unsqueeze(1)-forecasts, dim=-1)
    
    l1_col_mask = distances < threshold
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)

    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

def get_chomp_col_loss_ETH(plan, forecasts, neg_mask, threshold = 0.6, eps = 0.2):
    maskkk = neg_mask.clone()
    maskkk[maskkk == 0] = 999999999
    distances = torch.linalg.norm((plan.unsqueeze(1)-forecasts)*maskkk, dim=-1)

    # breakpoint()
    
    l1_col_mask = distances < threshold
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)

    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

def train_planner_forecaster_together(forecaster, planner, train_loader, criterion, forecaster_optimizer,\
          planner_optimizer, num_human=5, horizon=12, col_weight = 0.1, 
          col_threshold = 0.6, planning_horizon=12, device = 'cpu'):

    forecaster.train()
    planner.train()
    
    loss_sum_all, loss_sum_planner, loss_sum_forecaster, loss_sum_col = 0, 0, 0, 0

    loss_types = ['total_loss', 'mse', 'cost_dif', 'plan_cost', 'forecast_cost']

    loss_dict = {
        'planner': {},
        'forecaster': {}
    }
    for lt in loss_types:
        loss_dict['planner'][lt] = 0
        loss_dict['forecaster'][lt] = 0

    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch

        robot_states = robot_states.to(device)
        human_states = human_states.to(device)
        pos_seeds = pos_seeds.to(device)
        neg_seeds = neg_seeds.to(device)
        neg_hist = neg_hist.to(device)
        

        neg_hist = neg_hist.permute([0, 2, 1, 3])    
        neg_seeds = neg_seeds.permute([0, 2, 1, 3])    
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        ### Update planner
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()

        vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        plan_cost = get_chomp_col_loss(plan, forecasts)
        gt_cost = get_chomp_col_loss(pos_seeds, forecasts)
        
        planner_cost_dif = plan_cost - gt_cost
        planner_mse = criterion(plan, pos_seeds)
        planner_loss = planner_mse+col_weight*planner_cost_dif

        planner_loss.backward()
        planner_optimizer.step()

        losses = [planner_loss.item(), planner_mse.item(), 
        planner_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['planner'][lt] += loss

        ### Update Forecaster 
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()
        
        vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        plan_cost = get_chomp_col_loss(plan, forecasts)
        gt_cost = get_chomp_col_loss(pos_seeds, forecasts)
        
        forecaster_cost_dif = plan_cost - gt_cost
        forecaster_mse = criterion(forecasts, neg_seeds)
        forecaster_loss = forecaster_mse+col_weight*forecaster_cost_dif

        forecaster_loss.backward()
        forecaster_optimizer.step()
        
        losses = [forecaster_loss.item(), forecaster_mse.item(), 
        forecaster_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['forecaster'][lt] += loss

    num_batch = len(train_loader)
    for model in ['planner', 'forecaster']:
        for lt in loss_types:
            loss_dict[model][lt] /= num_batch
    return loss_dict

def train_planner_forecaster_together_validation(val_forecaster, planner, valid_loader, horizon=8, device = 'cpu'): 
    val_forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_gen_total, col_gt_total, cost_gen_total, cost_gt_total, total_datset_size = 0, 0, 0, 0, 0, 0 ,0
    for batch in valid_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist = batch

        robot_states = robot_states.to(device)
        human_states = human_states.to(device)
        pos_seeds = pos_seeds.to(device)
        neg_seeds = neg_seeds.to(device)
        neg_hist = neg_hist.to(device)

        neg_hist = neg_hist.permute([0, 2, 1, 3])    
        neg_seeds = neg_seeds.permute([0, 2, 1, 3])    
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        vel_forecasts, features = val_forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)
        ade, fde, col_gen, cost_gen = get_return(plan, pos_seeds, forecasts, horizon)
        _, _, col_gt, cost_gt = get_return(plan, pos_seeds, neg_seeds, horizon)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_gen_total += sum(col_gen)
        cost_gen_total += sum(cost_gen)
        col_gt_total += sum(col_gt)
        cost_gt_total += sum(cost_gt)
        total_datset_size += robot_states.shape[0]
    # breakpoint()
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_gen_total/total_datset_size, cost_gen_total/total_datset_size, col_gt_total/total_datset_size, cost_gt_total/total_datset_size



def train_planner_forecaster_together_ETH(forecaster, planner, train_loader, criterion, forecaster_optimizer,\
          planner_optimizer, num_human=5, horizon=12, col_weight = 0.1, 
          col_threshold = 0.6, planning_horizon=12, device = 'cpu'):

    forecaster.train()
    planner.train()
    
    loss_sum_all, loss_sum_planner, loss_sum_forecaster, loss_sum_col = 0, 0, 0, 0

    loss_types = ['total_loss', 'mse', 'cost_dif', 'plan_cost', 'forecast_cost']

    loss_dict = {
        'planner': {},
        'forecaster': {}
    }
    for lt in loss_types:
        loss_dict['planner'][lt] = 0
        loss_dict['forecaster'][lt] = 0
        
    for batch_data in tqdm.tqdm(train_loader, total=len(train_loader)):
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,60,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 
        ### Update planner
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()

        vel_forecasts, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)   
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        # breakpoint()
        plan_cost = get_chomp_col_loss_ETH(plan, forecasts, neg_mask)   # 여기 forecast는 로스 흐르지 않도록 설정해주자. 
        gt_cost = get_chomp_col_loss_ETH(pos_seeds.squeeze(1), forecasts, neg_mask)
        
        planner_cost_dif = plan_cost - gt_cost
        planner_mse = criterion(plan, pos_seeds.squeeze(1))
        planner_loss = planner_mse+col_weight*planner_cost_dif

        planner_loss.backward()
        planner_optimizer.step()

        losses = [planner_loss.item(), planner_mse.item(), 
        planner_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['planner'][lt] += loss

        ### Update Forecaster 
        '''
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()
        
        vel_forecasts, _ = forecaster(human_history, pad_mask_person)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)   
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성

        plan_cost = get_chomp_col_loss_ETH(plan, forecasts, neg_mask)
        gt_cost = get_chomp_col_loss_ETH(pos_seeds.squeeze(2), forecasts, neg_mask)
        
        forecaster_cost_dif = plan_cost - gt_cost
        forecaster_mse = criterion(forecasts*neg_mask, neg_seeds*neg_mask)
        forecaster_loss = forecaster_mse+col_weight*forecaster_cost_dif

        forecaster_loss.backward()
        forecaster_optimizer.step()
        
        losses = [forecaster_loss.item(), forecaster_mse.item(), 
        forecaster_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['forecaster'][lt] += loss
        '''
    num_batch = len(train_loader)
    # for model in ['planner', 'forecaster']:
    for model in ['planner']:
        for lt in loss_types:
            loss_dict[model][lt] /= num_batch
    return loss_dict



def train_planner_forecaster_together_validation_ETH(val_forecaster, planner, valid_loader, horizon=8, device = 'cpu'): 
    # 이거 맞는지 체크해야함. ADE, FDE 계산시 모든 forecast에 대해서 해주고 있음 
    val_forecaster.eval()
    planner.eval()

    ade_total, fde_total, col_total, jerk_total, miss_rate_total, total_datset_size = 0, 0, 0, 0, 0, 0
    for batch_data in tqdm.tqdm(valid_loader, total=len(valid_loader)):
        robot_states, human_states, action, _, _, _, _, neg_mask, pad_mask_person, _, robot_idx, _, _, neg_hist, pos_hist, neg_seeds, pos_seeds = batch_data 
        
        '''robot_states: Bx1x4, human_states: BxNx4, pos_seeds: BxTx1x2, neg_seeds: BxTxNx2, pos_hist: BxTx1x2, neg_hist: BxTxNx2, neg_mask: BxTxNx2, pad_mask_person: BxN'''
        robot_states, human_states, action, pos_seeds, neg_seeds, pos_hist, neg_hist, neg_mask, pad_mask_person = robot_states.cuda(), human_states.cuda(), action.cuda(), pos_seeds.cuda(), neg_seeds.cuda(), pos_hist.cuda(), neg_hist.cuda(), neg_mask.cuda(), pad_mask_person.cuda()
        neg_seeds = neg_seeds.permute(0,2,1,3)
        neg_hist = neg_hist.permute(0,2,1,3)
        human_history = neg_hist   # 128, 8, 60, 2
        human_states = neg_hist[:,-1,:,:]  # 128,60,2
        robot_states = pos_hist.squeeze()[:,-1,:]  # 128,60,2
        robot_states = torch.cat([robot_states, pos_hist.squeeze()[:, -1] - pos_hist.squeeze()[:, -2]], dim = -1) 

        with torch.no_grad():
            vel_forecasts, _ = val_forecaster(human_history, pad_mask_person)
            forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
            vel_plan = planner(robot_states, human_history, pad_mask_person, forecasts*neg_mask)   
            plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        ade, fde, col_gt, jerk, miss_rate = get_return_total(neg_mask = neg_mask, plans_sampled = plan, plans_gt = pos_seeds.squeeze(1), human_y_gt = neg_seeds, plan_horizon = pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
        ade_total += sum(ade)
        fde_total += sum(fde)
        col_total += sum(col_gt)
        jerk_total += sum(jerk)
        miss_rate_total += sum(miss_rate)
        total_datset_size += robot_states.shape[0]


    # breakpoint()
        
    return ade_total/total_datset_size, fde_total/total_datset_size, col_total/total_datset_size, jerk_total/total_datset_size, miss_rate_total/total_datset_size
