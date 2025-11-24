import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def viz_trajectory(pred=None, gt=None, prev_gt=None, output_dir=None, batch_idx=None, agent_idx_=None, epoch_number = 0):
    # pred: B, N, F_T, 2
    # gt: B, N, F_T, 2
    # prev_gt: B, N, H_T, 2
    
    if agent_idx_ is None: agent_idx_ = 0
    # breakpoint()
    B, N, T, XYZ_3 = pred.shape
    # if B == 1:
    #     return
    gt, pred = torch.cat((prev_gt[:,:,-1:], gt), dim=2), torch.cat((prev_gt[:,:,-1:], pred), dim=2)
    pred, gt, prev_gt = pred.cpu().detach().numpy(), gt.cpu().detach().numpy(), prev_gt.cpu().detach().numpy()
    # num_modes = pred.shape[0]
    # padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:inst_frame].cpu().numpy()
    # padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    # x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,-inst_frame:,0,:2].cpu().numpy()
    # y_gt = gt.reshape(B, N, T-1, 2).cpu().numpy()
    # y_pred = pred.reshape(num_modes, B, N, T-1, 2).cpu().detach().numpy()
    # valid_agents = (~(data.padding_mask.sum(-1) == data.padding_mask.shape[-1])).reshape(B, N)
    # min_idcs = min_idcs.reshape(B, N)
    # pred: 128, 60, 13, 2, gt: 128, 60, 13, 2
    draw_per_batch = 0
    if not os.path.isdir(output_dir): os.makedirs(output_dir)    
    for scene_idx in range(B):
        if draw_per_batch != 0:
            break
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        xy_mean = np.zeros((0, 2))
        for agent_idx in range(N):
            if (np.all(gt[scene_idx, agent_idx, :, 0] == 0) and np.all(prev_gt[scene_idx, agent_idx, :, 0] == 0)): 
                continue
            plot_style = 'ro-' 
            ax.plot(pred[scene_idx, agent_idx, :, 0], pred[scene_idx, agent_idx, :, 1], plot_style, linewidth=0.25, markersize=0.5)
            ax.plot(gt[scene_idx, agent_idx, :, 0], gt[scene_idx, agent_idx, :, 1], 'bo-', linewidth=0.25, markersize=0.5)
            ax.plot(prev_gt[scene_idx, agent_idx, :, 0], prev_gt[scene_idx, agent_idx, :, 1], 'ko-', linewidth=0.25, markersize=0.5)
            xy_mean = np.concatenate((xy_mean, prev_gt[scene_idx, agent_idx][:,:2]), axis=0)
            xy_mean = np.concatenate((xy_mean, gt[scene_idx, agent_idx][:,:2]), axis=0)

        if len(xy_mean) == 0:
            return
        ax.set_xlim([xy_mean.min(0)[0]-1, xy_mean.max(0)[0]+1])
        ax.set_ylim([xy_mean.min(0)[1]-1, xy_mean.max(0)[1]+1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(f'{output_dir}/epoch_{epoch_number}batch_{batch_idx}_scene_{scene_idx}.png', dpi=300)
        plt.close()
        plt.cla()
        draw_per_batch += 1
    # import pdb;pdb.set_trace()
    
def viz_trajectory_manyforecast(pred, gt, prev_gt, output_dir, batch_idx, agent_idx_=None, epoch_number = 0):
        # pred: B, N, T, 2
    # breakpoint()
    if agent_idx_ is None: agent_idx_ = 0
    B, N, num_modes, T, XYZ_3 = pred.shape
    # breakpoint()
    # gt, pred = torch.cat((prev_gt[:,:,-1:], gt), dim=2), torch.cat((prev_gt[:,:,-1:], pred), dim=2)
    pred, gt, prev_gt = pred.cpu().detach().numpy(), gt.cpu().detach().numpy(), prev_gt.cpu().detach().numpy()
    # num_modes = pred.shape[0]
    # padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:inst_frame].cpu().numpy()
    # padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    # x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,-inst_frame:,0,:2].cpu().numpy()
    # y_gt = gt.reshape(B, N, T-1, 2).cpu().numpy()
    # y_pred = pred.reshape(num_modes, B, N, T-1, 2).cpu().detach().numpy()
    # valid_agents = (~(data.padding_mask.sum(-1) == data.padding_mask.shape[-1])).reshape(B, N)
    # min_idcs = min_idcs.reshape(B, N)
    draw_per_batch = 0
    if not os.path.isdir(output_dir): os.makedirs(output_dir)    
    for scene_idx in range(B):
        if draw_per_batch != 0:
            break
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        xy_mean = np.zeros((0, 2))
        for agent_idx in range(N):
            if (np.all(gt[scene_idx, agent_idx, :, 0] == 0) and np.all(prev_gt[scene_idx, agent_idx, :, 0] == 0)): 
                continue
            plot_style = 'ro-' 
            for forecast_number in range(pred.shape[2]):
                ax.plot(pred[scene_idx, agent_idx, forecast_number, :, 0], pred[scene_idx, agent_idx, forecast_number, :, 1], plot_style, linewidth=0.25, markersize=0.5)
            ax.plot(gt[scene_idx, agent_idx, :, 0], gt[scene_idx, agent_idx, :, 1], 'bo-', linewidth=0.25, markersize=0.5)
            ax.plot(prev_gt[scene_idx, agent_idx, :, 0], prev_gt[scene_idx, agent_idx, :, 1], 'ko-', linewidth=0.25, markersize=0.5)
            xy_mean = np.concatenate((xy_mean, prev_gt[scene_idx, agent_idx][:,:2]), axis=0)
            xy_mean = np.concatenate((xy_mean, gt[scene_idx, agent_idx][:,:2]), axis=0)

        ax.set_xlim([xy_mean.min(0)[0]-1, xy_mean.max(0)[0]+1])
        ax.set_ylim([xy_mean.min(0)[1]-1, xy_mean.max(0)[1]+1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(f'{output_dir}/epoch_{epoch_number}batch_{batch_idx}_scene_{scene_idx}.png', dpi=300)
        plt.close()
        plt.cla()
        draw_per_batch += 1
    # import pdb;pdb.set_trace()