import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from argparse import ArgumentParser
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pathlib

# NOTE: import datamodule
from dataset.crowd_nav.datamodule_crowdnav import DataModule_CrowdNav
from dataset.ETH.datamodule_eth import DataModule_ETH  
from dataset.SIT.SIT_datamodule import SITDatamodule 
from dataset.MIX.MIX_datamodule import MIXDatamodule
from dataset.THOR.THOR_datamodule import THORDatamodule

# NOTE: import training planner code
from utils.training_planner import train_planner_uni        

# NOTE: import model
from policy.forecaster_model_uni import ForecasterModel_uni
from policy.planner_model_uni import PlannerModel_uni                 

# NOTE: setting seed to reproducing
from utils.reproduce_utils import set_seed

def main(args, cfg):
    HISTORY, HORIZON = cfg['model']['history'], cfg['model']['horizon']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.trained_dataset_name == "MIX":      # mix module의 경우 저장명을 바꿈.
        args.save = f'{args.save}_except_{args.except_dataset_name}'
        forecaster_mix_path = f'_except_{args.except_dataset_name}'
        planner_mix_path = f'_except_{args.except_dataset_name}'
    else:
        forecaster_mix_path = ""
        planner_mix_path = ""

    # NOTE: data loader    
    datamodule = {'crowd_nav': DataModule_CrowdNav,
                  'eth': DataModule_ETH,
                  'hotel': DataModule_ETH,
                  'univ': DataModule_ETH,
                  'zara1': DataModule_ETH,
                  'zara2': DataModule_ETH,
                  'THOR': THORDatamodule, 
                  'SIT': SITDatamodule,  
                  'MIX': MIXDatamodule,     
                  }[args.finetune_dataset_name](args, cfg)
    train_loader, valid_loader = datamodule.return_train_dataloader(), datamodule.return_valid_dataloader()

    # NOTE: load pretrained forecaster 
    forecaster = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
    forecaster_dict = torch.load(f"results/APTP/planner_merger/{args.trained_dataset_name}{forecaster_mix_path}{args.planner_exp_name}/Merging_at_{args.trained_dataset_name}/ckpt/merged_forecaster/best_ade.pth") 
    forecaster.load_state_dict(forecaster_dict)
    forecaster.eval()       # NOTE: planner 학습동안 forecaster는 학습되지 않음. Task vector와 관련해서는 좀 생각해봐야할 듯
    forecaster.to(device)
    
    # NOTE: planner model 
    torch.manual_seed(cfg["seed"])  
    planner = PlannerModel_uni(horizon=HORIZON, history = HISTORY)  
    planner_dict = torch.load(f"results/APTP/planner_merger/{args.trained_dataset_name}{planner_mix_path}{args.planner_exp_name}/Merging_at_{args.trained_dataset_name}/ckpt/merged_planner/best_ade.pth")
    planner.load_state_dict(planner_dict)
    planner.train()
    planner.to(device)

    # NOTE: model save at inital point
    os.makedirs(f'{args.save}/{args.exp_name}/ckpt', exist_ok=True)
    torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/initial_weight.pth')    
    
    # NOTE: optimizer, scheduler, loss function
    param = list(planner.parameters()) 
    planner_optimizer = optim.Adam(param, lr=1e-4)      # lr은 1e-4로 고정
    planner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        planner_optimizer, 'min', patience=20, threshold=0.01,
        factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    
    # NOTE: create logger
    if args.logger == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        # import datetime
        # log_dir=f'{args.save}/logs/{args.exp_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        log_dir=f'{args.save}/{args.exp_name}/logs'
        viz_dir=f'{args.save}/{args.exp_name}/viz'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)
        logger.viz_dir = viz_dir
    elif args.logger == 'none':
        logger = None
    else:
        raise KeyError(f'logger type {args.logger} is not defined')
    
    # NOTE: training process
    current_best_ade, current_best_jerk, current_best_col, current_best_fde, current_best_missrate = 99999999, 99999999, 99999999, 99999999, 99999999
    for i in range(args.max_epochs):
        if (i+1) % 1 == 0: print(f'Epoch {i+1}')
        train_loss_planner, _, _ = \
                        train_planner_uni(forecaster, planner, train_loader, criterion, planner_optimizer, \
                                            history=HISTORY, horizon=HORIZON, 
                                            col_weight=0.1, 
                                            is_train=True,      # True
                                            device = device,
                                            viz_dir = None,     # None
                                            epoch_number = None,
                                            args=args)

        planner_scheduler.step(train_loss_planner)
        print(f'Planner = {round(train_loss_planner, 3)}')
        logger.add_scalar('Train Loss Planner', train_loss_planner, i)
        torch.cuda.empty_cache()
        
        if (i+1) % 1 == 0:
            ade, fde, col, jerk, missrate, jerkrate = \
                        train_planner_uni(forecaster, planner, valid_loader, criterion, planner_optimizer, \
                                            history=HISTORY, horizon=HORIZON, 
                                            col_weight=0.1, 
                                            is_train=False,      # True
                                            device = device,
                                            viz_dir = viz_dir,     
                                            epoch_number = i+1,
                                            args=args)
        
            print(f"Epoch:{i}, ade: {ade}, fde: {fde}, col: {col}, jerk: {jerk}, missrate: {missrate}, jerkrate: {jerkrate}")
            logger.add_scalar('val/ade', ade, i)
            logger.add_scalar('val/fde', fde, i)
            logger.add_scalar('val/col', col, i)
            logger.add_scalar('val/jerk', jerk, i)
            logger.add_scalar('val/missrate', missrate, i)
            logger.add_scalar('val/jerkrate', jerkrate, i)
            logger.flush()
            torch.cuda.empty_cache()

        if current_best_ade > ade:
            current_best_ade = ade
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/best_ade.pth')
        if current_best_jerk > jerk:
            current_best_jerk = jerk
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/best_jerk.pth')
        if current_best_col > col:
            current_best_col = col
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/best_col_th06.pth')
        if current_best_fde > fde:
            current_best_fde = fde
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/best_fde.pth')
        if current_best_missrate > missrate:
            current_best_missrate = missrate
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/best_missrate_th05.pth')

        # ------- TODO: task vector 추출을 위해서 모델의 각 부분 모두를 epoch마다 따로 저장하도록 ------ # 
        torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/{i+1}.pth') 



        # --------------------------------------------------------------------------------- #


    logger.close()
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint for predplan training')
    parser.add_argument("--logger", type=str, default='none', choices=['none', 'wandb', 'tensorboard'])
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--planner_exp_name", type=str, required=True)
    parser.add_argument("--viz_interv", type=int, default=400)
    parser.add_argument("--trained_dataset_name", type=str, default=None)
    parser.add_argument("--finetune_dataset_name", type=str, default=None)
    parser.add_argument("--except_dataset_name", type=str, default=None)
    parser.add_argument("--using_human_as_robot", type=bool, default=False, help='for SIT, JRDB dataset') 
    parser.add_argument("--exp_with_specific_scene", type=str, default="No", help='for SIT, JRDB dataset') 
    parser.add_argument("--exp_scene_name", type=str, default=None)
    
    # ------------------ TODO: only change here!! ------------------ #
    # NOTE: "crowd_nav", "eth", "hotel", "univ", "zara1", "zara2", "SIT", "JRDB"
    # trained_dataset_name = "crowd_nav"
    # trained_dataset_name = "eth"
    # trained_dataset_name = "hotel"
    # trained_dataset_name = "univ"
    # trained_dataset_name = "zara1"
    # trained_dataset_name = "zara2"
    # trained_dataset_name = "THOR"           # change_here
    trained_dataset_name = "SIT"   
    # trained_dataset_name = "JRDB"

    # trained_dataset_name = "MIX"
    except_dataset_name = "SIT"       
    planner_exp_name = "__forecaster_best_ade__planner_best_ade"
    
    finetune_dataset_name = "SIT"

    exp_name = "20250210"

    # TODO: using specific scene in SIT (아직 랜덤하게 샘플링은 구현 안했음.)
    exp_with_specific_scene = "No"  # NOTE: if True, using specific scene in SIT dataset
    exp_scene_name = ""        # NOTE: Cafeteria, Corridor, Courtyard, Hallway, Lobby, Outdoor_Alley, Three_way_intersection  # Crossroad랑 Subway_Entrance은 val set에 X 
    # ------------------ ------------------------ ------------------ #

    args = parser.parse_args(f'--config configs/baseline_gametheoretic_{finetune_dataset_name}.yaml \
--save results/APTP/planner_merger_finetuned/{trained_dataset_name}_finetune_in_{finetune_dataset_name} --planner_exp_name {planner_exp_name} --exp_name {exp_name} --trained_dataset_name {trained_dataset_name} --finetune_dataset_name {finetune_dataset_name} \
--logger tensorboard --except_dataset_name {except_dataset_name} --exp_with_specific_scene {exp_with_specific_scene} --exp_scene_name {exp_scene_name}'.split(' '))  
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    set_seed(cfg["seed"])
    main(args, cfg)