import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
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
                  }[args.test_dataset_name](args, cfg)
    test_loader = datamodule.return_test_dataloader()

    # NOTE: load pretrained forecaster 
    forecaster = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
    forecaster_dict = torch.load(f"{args.forecaster_ckpt}") 
    forecaster.load_state_dict(forecaster_dict)
    forecaster.eval()     
    forecaster.to(device)
    
    # NOTE: load pretrained planner
    planner = PlannerModel_uni(horizon=HORIZON, history = HISTORY)  
    planner_dict = torch.load(f"{args.planner_ckpt}") 
    planner.load_state_dict(planner_dict)
    planner.eval()
    planner.to(device)

    ade, fde, col, jerk, missrate, jerkrate = \
                train_planner_uni(forecaster, planner, test_loader, None, None, \
                                    history=HISTORY, horizon=HORIZON, 
                                    col_weight=0.1, 
                                    is_train=False,      
                                    device=device,
                                    viz_dir=None,     
                                    epoch_number=0,
                                    args=args)

    print(f"Train: {args.trained_dataset_name}, Test: {args.test_dataset_name_str} // ade: {ade}, fde: {fde}, col: {col}, jerk: {jerk}, missrate: {missrate}, jerkrate: {jerkrate}")
    torch.cuda.empty_cache()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--planner_exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint for predplan training')
    parser.add_argument("--logger", type=str, default='none', choices=['none', 'wandb', 'tensorboard'])
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--viz_interv", type=int, default=400)
    parser.add_argument("--trained_dataset_name", type=str, default=None)
    parser.add_argument('--except_dataset_name', type=str, nargs="+", help='실제 학습 시 mix에서 제외할 데이터셋', default=None)
    parser.add_argument("--using_human_as_robot", type=bool, default=False, help='for SIT, JRDB dataset') 
    parser.add_argument("--test_dataset_name", type=str, default=None) 
    parser.add_argument("--test_dataset_name_str", type=str, default=None) 
    parser.add_argument("--exp_with_specific_scene", type=bool, default=False, help='for SIT, JRDB dataset') 
    parser.add_argument("--exp_scene_name", type=str, default=None)
    parser.add_argument("--forecaster_ckpt", type=str, default=None)
    parser.add_argument("--planner_ckpt", type=str, default=None)
    
    
    # NOTE: Target_domain
    full_domain = ["crowd_nav", "eth", "hotel", "univ", "zara1", "zara2", "THOR", "SIT"]    # fixed
    # TODO: Only change here ------------------------------------------------------------------------ #
    Target_domain = ["SIT"]         # ["THOR", "SIT"]    순서 꼭 지키기                                                    
    # ----------------------------------------------------------------------------------------------- #
    Source_domain = [domain for domain in full_domain if domain not in Target_domain]

    test_dataset_name = "MIX"
    except_dataset_name = ' '.join(Source_domain)
    test_dataset_name_str = '_'.join(Target_domain)
    train_dataset_name_str = '_'.join(Source_domain)

    planner_exp_name = "/20250129" 


    # TODO: 평가에 사용할 forecaster ckpt, planner ckpt

    # ------------------ NOTE: GameTheoretic ------------------ #
    # SIT dataset   -> Target_domain = ["SIT"] 
    # NOTE: 0. ZeroShot 성능
    # forecaster_ckpt = "results/APTP/forecaster/MIX_except_SIT/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/MIX_except_SIT/20250210_final/ckpt/best_ade.pth"

    # NOTE: 1. DA 성능
    # forecaster_ckpt = "results/APTP/forecaster_BASELINE/MIX_finetune_in_SIT_except_SIT/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_BASELINE/MIX_finetune_in_SIT_except_SIT/20250210/ckpt/best_ade.pth"

    # NOTE: 2. SCRACH 성능
    # forecaster_ckpt = "results/APTP/forecaster/SIT/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/SIT/20250210_final/ckpt/best_ade.pth"

    # NOTE: 3. ModelMerging: SIT__forecaster_best_ade__planner_best_ade
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade/Merging_at_SIT/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade/Merging_at_SIT/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade/Merging_at_SIT/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_SIT/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_SIT/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_SIT/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_SIT/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_SIT/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_SIT/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25
    # forecaster_ckpt = "/mnt/minseok/APTP/results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "/mnt/minseok/APTP/results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "/mnt/minseok/APTP/results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_SIT/ckpt/merged_planner_finetuned/best_ade.pth"


    # NOTE: 추가실험: 250304
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_planner_finetuned/best_ade.pth"

    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25/Merging_at_SIT_250304/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 추가실험: 250305
    # forecaster ade, 5, 10, 15
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner_finetuned/best_ade.pth"

    # forecaster 20 40 // 5, 10, 15
    # forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_20_40__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_20_40__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_20_40__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner_finetuned/best_ade.pth"

    # forecaster 30 60 // 5, 10, 15
    forecaster_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner/best_ade.pth"
    planner_ckpt = "results/APTP/planner_merger/SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15/Merging_at_SIT_250304/ckpt/merged_planner_finetuned/best_ade.pth"

    # THOR dataset   -> Target_domain = ["THOR"] 
    # NOTE: 0. ZeroShot 성능
    # forecaster_ckpt = "results/APTP/forecaster/MIX_except_THOR/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/MIX_except_THOR/20250210_final/ckpt/best_ade.pth"

    # NOTE: 1. DA 성능
    # forecaster_ckpt = "results/APTP/forecaster_BASELINE/MIX_finetune_in_THOR_except_THOR/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_BASELINE/MIX_finetune_in_THOR_except_THOR/20250210/ckpt/best_ade.pth"

    # NOTE: 2. SCRACH 성능
    # forecaster_ckpt = "results/APTP/forecaster/THOR/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/THOR/20250210_final/ckpt/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR__forecaster_best_ade__planner_best_ade
    # forecaster_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade/Merging_at_THOR/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade/Merging_at_THOR/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade/Merging_at_THOR/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05
    # forecaster_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_THOR/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_THOR/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05/Merging_at_THOR/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50
    # forecaster_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_THOR/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_THOR/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_THOR/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25
    # forecaster_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_planner_finetuned/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25
    # forecaster_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR/ckpt/merged_planner_finetuned/best_ade.pth"



    # SIT-THOR dataset  -> Target_domain = ["THOR", "SIT"] 
    # NOTE: 0. ZeroShot 성능
    # forecaster_ckpt = "results/APTP/forecaster/MIX_except_THOR_SIT/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/MIX_except_THOR_SIT/20250210_final/ckpt/best_ade.pth"

    # NOTE: 1. DA 성능
    # forecaster_ckpt = "results/APTP/forecaster_BASELINE/MIX_finetune_in_THOR_SIT_except_THOR_SIT/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_BASELINE/MIX_finetune_in_THOR_SIT_except_THOR_SIT/20250210/ckpt/best_ade.pth"

    # NOTE: 2. SCRACH 성능
    # forecaster_ckpt = "results/APTP/forecaster/MIX_except_crowd_nav_eth_hotel_univ_zara1_zara2/ckpt/best_ade.pth"
    # planner_ckpt = "results/APTP/planner/MIX_except_crowd_nav_eth_hotel_univ_zara1_zara2/20250210_final/ckpt/best_ade.pth"

    # NOTE: 3. ModelMerging: THOR_SIT
    # forecaster_ckpt = "results/APTP/planner_merger/THOR_SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR_SIT_debuggggggging_250216/ckpt/merged_forecaster/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR_SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR_SIT_debuggggggging_250216/ckpt/merged_planner/best_ade.pth"
    # planner_ckpt = "results/APTP/planner_merger/THOR_SIT__forecaster_best_ade_30_60__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_25/Merging_at_THOR_SIT_debuggggggging_250216/ckpt/merged_planner_finetuned/best_ade.pth"


    # --------------------------------------------------------- #




    # --------------------------------------------------------- #

    args = parser.parse_args(f'--config configs/baseline_gametheoretic_{test_dataset_name}.yaml \
--save results/APTP/planner/{train_dataset_name_str} --planner_exp_name {planner_exp_name} --trained_dataset_name {train_dataset_name_str} --test_dataset_name_str {test_dataset_name_str} \
--logger tensorboard --except_dataset_name {except_dataset_name} --test_dataset_name {test_dataset_name} \
--forecaster_ckpt {forecaster_ckpt} --planner_ckpt {planner_ckpt}'.split(' '))  
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    set_seed(cfg["seed"])
    main(args, cfg)