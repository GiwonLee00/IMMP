# TODO: 250209에 옮김. 변경사항 생기면 다시 수정.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# NOTE: import training forecaster code
from Game_utils.training_forecaster import train_forecaster_uni

# NOTE: import model
from Game_model.forecaster_model_uni import ForecasterModel_uni

# NOTE: setting seed to reproducing
from utils.reproduce_utils import set_seed

def main(args, cfg):
    HISTORY, HORIZON = cfg['model']['history'], cfg['model']['horizon']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset_name == "MIX":      # mix module의 경우 저장명을 바꿈.
        args.save = f'{args.save}_except_{args.except_dataset_name}'

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
                  }[args.dataset_name](args, cfg)
    train_loader, valid_loader = datamodule.return_train_dataloader(), datamodule.return_valid_dataloader()
    
    # NOTE: forecaster model 
    torch.manual_seed(cfg["seed"])      # model weight same init
    forecaster = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
    forecaster.to(device)

    # NOTE: model save at inital point
    os.makedirs(f'{args.save}/ckpt', exist_ok=True)
    torch.save(forecaster.state_dict(), f'{args.save}/ckpt/initial_weight.pth')    
    
    # NOTE: optimizer, scheduler, loss function
    param = list(forecaster.parameters()) 
    forecaster_optimizer = optim.Adam(param, lr=1e-3)        # lr은 1e-3로 고정
    forecaster_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        forecaster_optimizer, 'min', patience=20, threshold=0.01,
        factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    
    # NOTE: create logger
    if args.logger == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        # import datetime
        # log_dir=f'{args.save}/logs/{args.exp_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        log_dir=f'{args.save}/logs/{args.exp_name}'
        viz_dir=f'{args.save}/viz'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)
        logger.viz_dir = viz_dir
    elif args.logger == 'none':
        logger = None
    else:
        raise KeyError(f'logger type {args.logger} is not defined')

    # NOTE: training process
    current_best_ade = 9999999
    for i in range(args.max_epochs):
        if (i+1) % 1 == 0: print(f'Epoch {i+1}')
        train_loss_forecaster = \
                        train_forecaster_uni(forecaster, train_loader, criterion, forecaster_optimizer, \
                                                history=HISTORY, horizon=HORIZON, 
                                                col_weight=0.0, 
                                                is_train=True,      # True
                                                device = device,
                                                viz_dir = None,  # None
                                                epoch_number = None,
                                                args=args)

        forecaster_scheduler.step(train_loss_forecaster)
        print("train loss: ", f'{round(train_loss_forecaster, 5)}')
        logger.add_scalar('Train Loss Forecaster', train_loss_forecaster, i)
        
        if (i+1) % 1 == 0:
            ade, fde = train_forecaster_uni(forecaster, valid_loader, criterion, forecaster_optimizer, \
                                                history=HISTORY, horizon=HORIZON,
                                                col_weight=0.0, 
                                                is_train=False,      # False
                                                device = device,
                                                viz_dir = None,   # two_option: viz_dir, None 
                                                epoch_number = i+1,
                                                args=args)
                
            print(f"Epoch: {i}  ade: {ade}, fde: {fde}")
            logger.add_scalar("val/ade", ade, i)
            logger.add_scalar("val/fde", fde, i)
            logger.flush()

        if current_best_ade > ade:
            current_best_ade = ade
            torch.save(forecaster.state_dict(), f'{args.save}/ckpt/best_ade.pth')  # best ade만 저장

        # ------- TODO: task vector 추출을 위해서 모델의 각 부분 모두를 epoch마다 따로 저장하도록 ------ # 
        torch.save(forecaster.state_dict(), f'{args.save}/ckpt/{i+1}.pth') 

        # NOTE: 일단은 forecaster는 아직 task vector를 추출하지 않아도 되지 않을까??? 쩝... 아닌가 이것도 task vector 모두 추출해야할듯...

        # --------------------------------------------------------------------------------- #

    logger.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint for predplan training')
    parser.add_argument("--logger", type=str, default='none', choices=['none', 'wandb', 'tensorboard'])
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--viz_interv", type=int, default=400)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--except_dataset_name", type=str, default=None)
    parser.add_argument("--using_human_as_robot", type=bool, default=False, help='for SIT, JRDB dataset')   
    parser.add_argument("--exp_with_specific_scene", type=bool, default=False, help='for SIT, JRDB dataset') 
    parser.add_argument("--exp_scene_name", type=str, default=None) 

    # ------------------ TODO: only change here!! ------------------ #
    # NOTE: "crowd_nav", "eth", "hotel", "univ", "zara1", "zara2", "SIT", "JRDB"
    # dataset_name = "crowd_nav"
    # dataset_name = "eth"
    # dataset_name = "hotel"
    # dataset_name = "univ"
    # dataset_name = "zara1"
    # dataset_name = "zara2"
    # dataset_name = "THOR"
    dataset_name = "SIT"   
    # dataset_name = "JRDB"
    
    # dataset_name = "MIX"
    except_dataset_name = "SIT"        # dataset_name == "MIX" 일 때, 파일 저장명 바꾸기 위함. (아닐때는 그냥 암거나 가능)

    # ------------------ ------------------------ ------------------ #
    # TODO: save에 _debug를 추가해두었음. 원래는 빼고 함
    args = parser.parse_args(f'--config configs/baseline_gametheoretic_{dataset_name}.yaml \
--save results/APTP/forecaster/{dataset_name} --exp_name debug --dataset_name {dataset_name} \
--logger tensorboard --except_dataset_name {except_dataset_name}'.split(' '))  
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    set_seed(cfg["seed"])
    main(args, cfg)