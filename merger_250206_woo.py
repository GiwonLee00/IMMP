import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from argparse import ArgumentParser
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pathlib
import math

# NOTE: import datamodule
from dataset.crowd_nav.datamodule_crowdnav import DataModule_CrowdNav
from dataset.ETH.datamodule_eth import DataModule_ETH  
from dataset.SIT.SIT_datamodule import SITDatamodule 
from dataset.MIX.MIX_datamodule import MIXDatamodule
from dataset.THOR.THOR_datamodule import THORDatamodule

# NOTE: import training planner code (finetuning 전용)
from utils.training_planner import train_planner_uni  

# NOTE: import training planner code (merge 전용)
from utils.merge_phase import merge_phase

# NOTE: import model
from policy.forecaster_model_uni import ForecasterModel_uni
from policy.planner_model_uni import PlannerModel_uni                 
from merging_factory.model_merge import Collect_Vector_foreplan, Collect_Vector, Merged_Model

# NOTE: setting seed to reproducing
from utils.reproduce_utils import set_seed
# tensorboard --logdir /mnt/minseok/APTP/results/APTP/planner_merger/SIT__forecaster_best_ade__planner_best_ade_best_col_th06_best_fde_best_missrate_th05_5_10_15_20_25_30_35_40_45_50/Merging_at_SIT/merging_finetuning_logs --host 143.248.58.40 --port 6044

def main(args, cfg):
    HISTORY, HORIZON = cfg['model']['history'], cfg['model']['horizon']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NOTE: if using specific scene, change save file name.
    if args.exp_with_specific_scene == True:
        args.save = args.save + f"_scene_{args.exp_scene_name}"

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

    train_loader, valid_loader = datamodule.return_train_dataloader(), datamodule.return_valid_dataloader()
    test_loader = datamodule.return_test_dataloader()

    ##################
    # previous 
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade"]
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05"]
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "25"]
    # forecaster_sample, planner_sample = ["best_ade", "30", "60"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "25"]

    # current
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15", "20", "25"]
    # forecaster_sample, planner_sample = ["best_ade", "30", "60"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15", "20", "25"]

    # current
    # forecaster_sample, planner_sample = ["best_ade"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15"]
    # forecaster_sample, planner_sample = ["best_ade", "20", "40"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15"]
    forecaster_sample, planner_sample = ["best_ade", "30", "60"], ["best_ade", "best_col_th06", "best_fde", "best_missrate_th05", "5", "10", "15"]

    # TODO: forecaster에 plan을 쓸지말지 결정. True면 plan 정보도 함께 합치도록 함.
    foreplan = False
    if foreplan:
        init_vector_f, init_vector_p, forecaster_tvd, planner_tvd, forecaster_list, planner_list = Collect_Vector_foreplan(cfg, args, forecaster_sample, planner_sample)
        planner_sample_foreplan = [item + "_foreplan" for item in planner_sample]
        forecaster_sample = forecaster_sample + planner_sample_foreplan
    else:
        init_vector_f, init_vector_p, forecaster_tvd, planner_tvd, forecaster_list, planner_list = Collect_Vector(cfg, args, forecaster_sample, planner_sample)
    
    args.save = args.save + "__forecaster"
    for point in forecaster_sample:
        args.save = args.save + f"_{point}"
    args.save = args.save + "__planner"
    for point in planner_sample:
        args.save = args.save + f"_{point}"

    forecaster_spec = [len(forecaster_tvd.keys()), len(forecaster_sample), len(forecaster_list[0])]
    print("#"*20+" Forecaseter "+"#"*20)
    print(f"Number of Datasets: {len(forecaster_tvd.keys())} --- {[k for k in iter(forecaster_tvd.keys())]}")
    print(f"Number of Sample: {len(forecaster_sample)} --- {forecaster_sample}")
    print(f"Granularity Level: {len(forecaster_list[0])} --- {forecaster_list[0]}")
    planner_spec = [len(planner_tvd.keys()), len(planner_sample), len(planner_list[0])]
    print("#"*20+" Planner "+"#"*20)
    print(f"Number of Datasets: {len(planner_tvd.keys())} --- {[k for k in iter(planner_tvd.keys())]}")
    print(f"Number of Sample: {len(planner_sample)} --- {planner_sample}")
    print(f"Granularity Level: {len(planner_list[0])} --- {planner_list[0]}")

    forecaster = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
    planner = PlannerModel_uni(history=HISTORY, horizon=HORIZON)
    _forecaster = [forecaster, init_vector_f, forecaster_tvd, forecaster_spec, forecaster_list]
    _planner = [planner, init_vector_p, planner_tvd, planner_spec, planner_list]
    
    merged_model = Merged_Model(_forecaster, _planner, f_prior=1/math.prod(forecaster_spec), p_prior=1/math.prod(planner_spec))
    merged_model.cuda()

    ##################
    optimizer = optim.Adam(merged_model.collect_trainable_params(), lr=1e-3)        # 원래는 lr 1e-3
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, threshold=0.01,
            factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()    

    ##################
    # NOTE: create logger
    if args.logger == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        # import datetime
        # log_dir=f'{args.save}/logs/{args.exp_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        log_dir=f'{args.save}/{args.exp_name}/merging_logs'
        viz_dir=f'{args.save}/{args.exp_name}/merging_viz'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)
        logger.viz_dir = viz_dir
    elif args.logger == 'none':
        logger = None
    else:
        raise KeyError(f'logger type {args.logger} is not defined')
    
    ##################
    # NOTE: merging weight process
    current_best_ade, current_best_jerk, current_best_col, current_best_fde, current_best_missrate = 99999999, 99999999, 99999999, 99999999, 99999999
    current_train_best_ade = 99999999
    for i in range(args.max_epochs_merging):
        if (i+1) % 1 == 0: print(f'Epoch {i+1}')
        train_loss_planner, _, _, merged_model = \
                        merge_phase(is_train=True, cfg=cfg, args=args, train_loader=train_loader, device=device, viz_dir=viz_dir, epoch_number=i+1, col_weight=0.1, merged_model=merged_model, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        
        
        print(f'Planner = {round(train_loss_planner, 3)}')
        logger.add_scalar('Train Loss Planner', train_loss_planner, i)
        torch.cuda.empty_cache()
        os.makedirs(f'{args.save}/{args.exp_name}/ckpt/merged_forecaster', exist_ok=True)    
        os.makedirs(f'{args.save}/{args.exp_name}/ckpt/merged_planner', exist_ok=True)        
        if (i+1) % 1 == 0:      # just 평가용
            ade, fde, col, jerk, missrate, jerkrate = \
                merge_phase(is_train=False, cfg=cfg, args=args, train_loader=valid_loader, device=device, viz_dir=viz_dir, epoch_number=i+1, col_weight=0.1, merged_model=merged_model)
        
        
            print(f"Epoch:{i} Validation, ade: {ade}, fde: {fde}, col: {col}, jerk: {jerk}, missrate: {missrate}, jerkrate: {jerkrate}")
            logger.add_scalar('val/ade', ade, i)
            logger.add_scalar('val/fde', fde, i)
            logger.add_scalar('val/col', col, i)
            logger.add_scalar('val/jerk', jerk, i)
            logger.add_scalar('val/missrate', missrate, i)
            logger.add_scalar('val/jerkrate', jerkrate, i)
            logger.flush()

            if current_best_ade > ade:
                current_best_ade = ade
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_ade.pth', type='forecaster')
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_ade.pth', type='planner')
            if current_best_jerk > jerk:
                current_best_jerk = jerk
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_jerk.pth', type='forecaster')
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_jerk.pth', type='planner')
            if current_best_col > col:
                current_best_col = col
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_col_th06.pth', type='forecaster')
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_col_th06.pth', type='planner')
            if current_best_fde > fde:
                current_best_fde = fde
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_fde.pth', type='forecaster')
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_fde.pth', type='planner')
            if current_best_missrate > missrate:
                current_best_missrate = missrate
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_missrate_th05.pth', type='forecaster')
                merged_model.save_weights(dir=f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_missrate_th05.pth', type='planner')
        
    logger.close()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------- # 
    # NOTE: finetuning with merged weight process

    # NOTE: STEP1: load pretrained forecaster and planner model
    forecaster = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
    forecaster_dict = torch.load(f'{args.save}/{args.exp_name}/ckpt/merged_forecaster/best_ade.pth')          # train 
    forecaster.load_state_dict(forecaster_dict)
    forecaster.eval()       # GameTheretic에서는 forecaster 학습하지 않음. 
    forecaster.to(device)

    planner = PlannerModel_uni(horizon=HORIZON, history = HISTORY)  
    planner_dict = torch.load(f'{args.save}/{args.exp_name}/ckpt/merged_planner/best_ade.pth')
    planner.load_state_dict(planner_dict)
    planner.train()
    planner.to(device)

    os.makedirs(f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned', exist_ok=True)  
    # NOTE: STEP2: optimizer, scheduler, loss function
    param = list(planner.parameters()) 
    planner_optimizer = optim.Adam(param, lr=1e-5)      # lr은 1e-4로 고정
    planner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        planner_optimizer, 'min', patience=20, threshold=0.01,
        factor=0.5, cooldown=20, min_lr=1e-6, verbose=True) # 1e-5로 고정
    criterion = nn.MSELoss()

    # NOTE: create logger
    if args.logger == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        # import datetime
        # log_dir=f'{args.save}/logs/{args.exp_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        log_dir=f'{args.save}/{args.exp_name}/merging_finetuning_logs'
        viz_dir=f'{args.save}/{args.exp_name}/merging_finetuning_viz'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)
        logger.viz_dir = viz_dir
    elif args.logger == 'none':
        logger = None
    else:
        raise KeyError(f'logger type {args.logger} is not defined')

    current_best_ade, current_best_jerk, current_best_col, current_best_fde, current_best_missrate = 99999999, 99999999, 99999999, 99999999, 99999999
    for i in range(args.max_epochs_merging_finetuning):
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
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/best_ade.pth')
        if current_best_jerk > jerk:
            current_best_jerk = jerk
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/best_jerk.pth')
        if current_best_col > col:
            current_best_col = col
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/best_col_th06.pth')
        if current_best_fde > fde:
            current_best_fde = fde
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/best_fde.pth')
        if current_best_missrate > missrate:
            current_best_missrate = missrate
            torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/best_missrate_th05.pth')

        # ------- TODO: task vector 추출을 위해서 모델의 각 부분 모두를 epoch마다 따로 저장하도록 ------ # 
        torch.save(planner.state_dict(), f'{args.save}/{args.exp_name}/ckpt/merged_planner_finetuned/{i+1}.pth') 
    
    logger.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint for predplan training')
    parser.add_argument("--logger", type=str, default='none', choices=['none', 'wandb', 'tensorboard'])
    parser.add_argument("--max_epochs_merging", type=int, default=800)
    parser.add_argument("--max_epochs_merging_finetuning", type=int, default=200)
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--viz_interv", type=int, default=400)
    # parser.add_argument("--total_dataset_list", type=str, nargs='+', default=None, help="List of train+test datasets")
    parser.add_argument("--test_dataset_name", type=str, default=None)
    parser.add_argument("--using_human_as_robot", type=bool, default=False, help='for SIT, JRDB dataset') 
    parser.add_argument("--planner_exp_name", type=str, default=None)
    parser.add_argument("--exp_with_specific_scene", type=str, default="No", help='for SIT, JRDB dataset') 
    parser.add_argument("--exp_scene_name", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default='results/APTP', help='pretrained checkpoint path')
    parser.add_argument('--except_dataset_name', type=str, nargs="+", help='실제 학습 시 mix에서 제외할 데이터셋', default=None)

    # NOTE: Target_domain
    full_domain = ["crowd_nav", "eth", "hotel", "univ", "zara1", "zara2", "THOR", "SIT"]    # fixed
    # TODO: Only change here ------------------------------------------------------------------------ #
    Target_domain = ["SIT"]            # ["THOR", "SIT"]                                                         
    # ----------------------------------------------------------------------------------------------- #
    Source_domain = [domain for domain in full_domain if domain not in Target_domain]

    test_dataset_name = "MIX"
    except_dataset_name = ' '.join(Source_domain)
    test_dataset_name_str = '_'.join(Target_domain)

    # breakpoint()

    # TODO: merging 실험 이름 넣어주기.
    exp_name = f"Merging_at_{test_dataset_name_str}_250304"
    # TODO: planning 학습 시에 사용한 exp name을 입력. 만약 없으면 "".
    planner_exp_name = "/20250210_final"          # TODO: 여기 수정하기 
    
    # TODO: using specific scene in SIT (아직 랜덤하게 샘플링은 구현 안했음.)
    exp_with_specific_scene = "No"  # NOTE: if True, using specific scene in SIT dataset
    exp_scene_name = ""        # NOTE: Cafeteria, Corridor, Courtyard, Hallway, Lobby, Outdoor_Alley, Three_way_intersection  # Crossroad랑 Subway_Entrance은 val set에 X 
    # ------------------ ------------------------ ------------------ #
    
    args = parser.parse_args(f'--config configs/merger.yaml \
--save results/APTP/planner_merger/{test_dataset_name_str} --exp_name {exp_name} --test_dataset_name {test_dataset_name} \
--logger tensorboard --planner_exp_name {planner_exp_name} --except_dataset_name {except_dataset_name} --exp_with_specific_scene {exp_with_specific_scene} --exp_scene_name {exp_scene_name}'.split(" "))
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    set_seed(cfg["seed"])
    main(args, cfg)