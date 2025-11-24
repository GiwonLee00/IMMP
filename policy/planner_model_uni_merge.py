import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from policy.forecaster_model_uni import ForecasterModel_uni
from policy.planner_model_uni import PlannerModel_uni     

from merging_factory.task_vector_v2 import TaskVector
from merging_factory.weight_extracter import weight_extracter_forecaster, weight_extracter_planner

import torch.optim as optim
import pytorch_lightning as pl
import pathlib
from utils.viz_utils import *
from utils.metric import *  


def get_chomp_col_loss_ETH(plan, forecasts, neg_mask, threshold = 0.6, eps = 0.2):
    maskkk = neg_mask.clone()
    maskkk[maskkk == 0] = 999999999
    distances = torch.linalg.norm((plan.unsqueeze(1)-forecasts)*maskkk, dim=-1)
    
    l1_col_mask = distances < threshold
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)

    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

class TrainableWeights(nn.Module):
    def __init__(self, module_len, dataset_len, forecaster_sampling_epoch_len, planner_sampling_epoch_len, device):
        super(TrainableWeights, self).__init__()
        self.forecaster_trainable_weights = nn.Parameter(
            torch.randn(module_len, dataset_len, forecaster_sampling_epoch_len, device=device, requires_grad=True)
        )
        self.planner_trainable_weights = nn.Parameter(
            torch.randn(module_len, dataset_len, planner_sampling_epoch_len, device=device, requires_grad=True)
        )

class PlannerModel_uni_merge(nn.Module):
    """ Policy network for imitation learning """
    def __init__(self, device=None, cfg=None, args=None):
        super().__init__()

        self.cfg = cfg
        self.args=args
        HISTORY, HORIZON = self.cfg['model']['history'], self.cfg['model']['horizon']
        # torch.manual_seed(cfg["seed"])  
        self.forecaster_pretrained = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
        self.forecaster_pretrained.to(device)
        # torch.manual_seed(cfg["seed"])  
        self.planner_pretrained = PlannerModel_uni(history=HISTORY, horizon=HORIZON)
        self.planner_pretrained.to(device)

        self.forecaster_new = ForecasterModel_uni(history=HISTORY, horizon=HORIZON)
        self.forecaster_new.to(device)

        self.planner_new = PlannerModel_uni(history=HISTORY, horizon=HORIZON)
        self.planner_new.to(device)

        # TODO: 적절한 값으로 변경해주기
        module_len = 3
        dataset_len = len(cfg["total_dataset_list"]) - 1
        forecaster_sampling_epoch_len = 1 # len(forecaster_sampling_epoch)
        planner_sampling_epoch_len = 1 # len(planner_sampling_epoch)

        self.trainable_weights = TrainableWeights(
            module_len=module_len,
            dataset_len=dataset_len,
            forecaster_sampling_epoch_len=forecaster_sampling_epoch_len,
            planner_sampling_epoch_len=planner_sampling_epoch_len,
            device=device
        )

        self.forecaster_task_vector_list = []
        self.forecaster_task_vector_name_dict = {}
        self.planner_task_vector_list = []
        self.planner_task_vector_name_dict = {}
        current_forecaster_idx = 0
        current_planner_idx = 0
        # pretrained model을 로드하고 파라미터 리스트를 추출
        for merging_dataset in cfg["total_dataset_list"]:
            if merging_dataset == args.test_dataset_name:
                continue
            # NOTE: load initial weight, and extract weight per module
            merging_forecaster_start = torch.load(f"/mnt/minseok/APTP/results/APTP/forecaster/{merging_dataset}/ckpt/initial_weight.pth")
            merging_planner_start = torch.load(f"/mnt/minseok/APTP/results/APTP/planner/{merging_dataset}{args.planner_exp_name}/ckpt/initial_weight.pth")
            self.forecaster_pretrained.load_state_dict(merging_forecaster_start, strict=True)
            self.planner_pretrained.load_state_dict(merging_planner_start, strict=True)

            """
            이렇게 하면 값이 바뀌어버림
            merging_forecaster_start = list(self.forecaster_pretrained.parameters()) 
            forecaster_start_keys_list = list(self.forecaster_pretrained.state_dict().keys())
            merging_planner_start = list(self.planner_pretrained.parameters()) 
            planner_start_keys_list = list(self.planner_pretrained.state_dict().keys())
            """
            merging_forecaster_start = [p.clone().detach() for p in self.forecaster_pretrained.parameters()]
            forecaster_start_keys_list = list(self.forecaster_pretrained.state_dict().keys())
            merging_planner_start = [p.clone().detach() for p in self.planner_pretrained.parameters()]
            planner_start_keys_list = list(self.planner_pretrained.state_dict().keys())

            merging_forecaster_start, self.forecaster_module_name_list, self.forecaster_module_name_detail_list = weight_extracter_forecaster(weight=merging_forecaster_start, keys_list=forecaster_start_keys_list)
            merging_planner_start, self.planner_module_name_list, self.planner_module_name_detail_list = weight_extracter_planner(weight=merging_planner_start, keys_list=planner_start_keys_list)

            # NOTE: load end weight, and extract weight per module
            forecaster_sampling_epoch = ["best_ade"]
            planner_sampling_epoch = ["best_ade"] # 여기도 best_ade, best jerk 이런거 값으로 넣어줘도 좋을듯?
            for sampling_epoch in forecaster_sampling_epoch:   
                merging_forecaster_end = torch.load(f"/mnt/minseok/APTP/results/APTP/forecaster/{merging_dataset}/ckpt/{sampling_epoch}.pth")   
                # breakpoint()
                self.forecaster_pretrained.load_state_dict(merging_forecaster_end, strict=True)
                
                """
                merging_forecaster_end = list(self.forecaster_pretrained.parameters())  
                forecaster_end_keys_list = list(self.forecaster_pretrained.state_dict().keys())
                """
                merging_forecaster_end = [p.clone().detach() for p in self.forecaster_pretrained.parameters()]
                forecaster_end_keys_list = list(self.forecaster_pretrained.state_dict().keys())

                
                merging_forecaster_end, _, _ = weight_extracter_forecaster(weight=merging_forecaster_end, keys_list=forecaster_end_keys_list)
                
                for module_forecaster_start, module_forecaster_end, module_forecaster_name in zip(merging_forecaster_start, merging_forecaster_end, self.forecaster_module_name_list):   
                    forecaster_vector = TaskVector(pretrained_checkpoint=module_forecaster_start, finetuned_checkpoint=module_forecaster_end, vector_module_name=module_forecaster_name, vector_dataset_name=merging_dataset, vector_epoch_name=sampling_epoch)
                    # breakpoint()
                    self.forecaster_task_vector_list.append(forecaster_vector)
                    self.forecaster_task_vector_name_dict.setdefault(module_forecaster_name, {}).setdefault(merging_dataset, {})[str(sampling_epoch)] = current_forecaster_idx
      
                    current_forecaster_idx += 1
                    # forecaster_task_vector_dict.setdefault(module_forecaster_name, {}).setdefault(merging_dataset, {})[str(sampling_epoch)] = forecaster_vector
            for sampling_epoch in planner_sampling_epoch:    
                merging_planner_end = torch.load(f"/mnt/minseok/APTP/results/APTP/planner/{merging_dataset}{args.planner_exp_name}/ckpt/{sampling_epoch}.pth")
                self.planner_pretrained.load_state_dict(merging_planner_end, strict=True)
                """
                merging_planner_end = list(self.planner_pretrained.parameters())  
                planner_end_keys_list = list(self.planner_pretrained.state_dict().keys())
                """
                merging_planner_end = [p.clone().detach() for p in self.planner_pretrained.parameters()]
                planner_end_keys_list = list(self.planner_pretrained.state_dict().keys())

                merging_planner_end, _, _ = weight_extracter_planner(weight=merging_planner_end, keys_list=planner_end_keys_list)
                
                for module_planner_start, module_planner_end, module_planner_name in zip(merging_planner_start, merging_planner_end, self.planner_module_name_list):
                    planner_vector = TaskVector(pretrained_checkpoint=module_planner_start, finetuned_checkpoint=module_planner_end, vector_module_name=module_planner_name, vector_dataset_name=merging_dataset, vector_epoch_name=sampling_epoch)
                    self.planner_task_vector_list.append(planner_vector)
                    self.planner_task_vector_name_dict.setdefault(module_planner_name, {}).setdefault(merging_dataset, {})[str(sampling_epoch)] = current_planner_idx
                    current_planner_idx += 1

        self.param = [{'params': self.trainable_weights.parameters()}]
        
        # breakpoint()
        self.optimizer = optim.Adam(self.param, lr=1e-3)      # lr은 merging에서는 lr을 크게 잡는다.
        self.planner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=20, threshold=0.01,
            factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)
        self.criterion = nn.MSELoss()


        # 1. pretrained parameter를 불러와서 리스트로 추출해서, self. 뭐시기로 저장해둠.
        # 2. trainable weight를 여기서 정의해줌. -> 이 또한 self. 뭐시기로 저장해둠.
    def set_attr(self, obj, names, val):
        if len(names) == 1:
            # breakpoint()
            setattr(obj, names[0], val)
            # cur = getattr(obj, names[0])
            # cur.data = val
        else:
            self.set_attr(getattr(obj, names[0]), names[1:], val)

    def load_weights(self, mod, names, params):
        # breakpoint()
        
        # for name, p in zip(names, params):
        self.set_attr(mod, names.split("."), params)

    def forward(self, batch_data, batch_idx, viz_dir, epoch_number, col_weight, is_train):   # crowd_obsv = human_history!!
        if is_train:
            self.forecaster_new.train()      # merging에서는 forecaster와 planner가 둘 다 들어감.
            self.planner_new.train()
        else:
            self.forecaster_new.eval()
            self.planner_new.eval()
        forecaster_weighted_sum_module_vector = []
        forecaster_weighted_sum_module_vector_name_dict = {}
        forcaster_module_cur_idx = 0
        for i, merging_module in enumerate(self.forecaster_task_vector_name_dict.keys()):
            cur_module_vector = 0
            for j, merging_dataset in enumerate(self.forecaster_task_vector_name_dict[merging_module].keys()):
                for k, merging_epoch in enumerate(self.forecaster_task_vector_name_dict[merging_module][merging_dataset].keys()):     
                    cur_module_vector += self.forecaster_task_vector_list[self.forecaster_task_vector_name_dict[merging_module][merging_dataset][merging_epoch]] * self.trainable_weights.forecaster_trainable_weights[i, j, k]
                    if j == 1: breakpoint()
                    # breakpoint()
                    # print("\n".join(f"{name}: requires_grad = {param.requires_grad}" for name, param in self.cur_module_vector.named_parameters()))
            # breakpoint()
            forecaster_weighted_sum_module_vector.append(cur_module_vector)
            forecaster_weighted_sum_module_vector_name_dict[merging_module] = forcaster_module_cur_idx
            forcaster_module_cur_idx += 1

        planner_weighted_sum_module_vector = []
        planner_weighted_sum_module_vector_name_dict = {}
        planner_module_cur_idx = 0
        for i, merging_module in enumerate(self.planner_task_vector_name_dict.keys()):
            cur_module_vector = 0
            for j, merging_dataset in enumerate(self.planner_task_vector_name_dict[merging_module].keys()):
                for k, merging_epoch in enumerate(self.planner_task_vector_name_dict[merging_module][merging_dataset].keys()):
                    cur_module_vector += self.planner_task_vector_list[self.planner_task_vector_name_dict[merging_module][merging_dataset][merging_epoch]] * self.trainable_weights.planner_trainable_weights[i, j, k]

            # planner_weighted_sum_module_vector[merging_module] = cur_module_vector        
            planner_weighted_sum_module_vector.append(cur_module_vector)
            planner_weighted_sum_module_vector_name_dict[merging_module] = planner_module_cur_idx
            planner_module_cur_idx += 1

        merging_forecaster_start = list(self.forecaster_new.parameters())  
        forecaster_start_keys_list = list(self.forecaster_new.state_dict().keys())
        merging_planner_start = list(self.planner_new.parameters())  
        planner_start_keys_list = list(self.planner_new.state_dict().keys())

        train_forecaster = True     # TODO: if True, find forecaster weight by merging method. If False, using SIT forcaster
        if train_forecaster == True:
            forecaster_state_dict, _, _ = weight_extracter_forecaster(weight=merging_forecaster_start, keys_list=forecaster_start_keys_list)
            for m, (module_name, detail_module_name) in enumerate(zip(self.forecaster_module_name_list, self.forecaster_module_name_detail_list)): 
                # breakpoint()
                cur_weight = forecaster_weighted_sum_module_vector[forecaster_weighted_sum_module_vector_name_dict[module_name]].vector
                # breakpoint()
                # cur_weight = forecaster_weighted_sum_module_vector[forecaster_weighted_sum_module_vector_name_dict[module_name]].apply_to(forecaster_state_dict[m])
                for name, param in self.forecaster_new.named_parameters():
                    # breakpoint()
                    
                    if name in detail_module_name:
                        name_number = detail_module_name.index(name)
                        # breakpoint()
           
                        if len(name.split(".")) == 2:
                            param = getattr(getattr(self.forecaster_new, name.split(".")[0]), name.split(".")[1])  # model.linear.weight에 접근
                        elif len(name.split(".")) == 3:
                            param = getattr(getattr(getattr(self.forecaster_new, name.split(".")[0]), name.split(".")[1]), name.split(".")[2])
                        
                        param.data += cur_weight[name_number]
             
                        
                        # self.load_weights(self.forecaster_new, name, cur_weight[name_number])
                        
        else:
            raise ValueError("아직 구현 X")
            forecaster_dict = torch.load(f"/mnt/minseok/APTP/results/APTP/forecaster/{args.test_dataset_name}/ckpt/best_ade.pth") 
            for name, param in self.forecaster_new.named_parameters():
                if name in cur_weight:
                    self.forecaster_new._parameters[name] = cur_weight[name]

        planner_state_dict, _, _ = weight_extracter_planner(weight=merging_planner_start, keys_list=planner_start_keys_list)
        for m, (module_name, detail_module_name) in enumerate(zip(self.planner_module_name_list, self.planner_module_name_detail_list)): 
            # breakpoint()
            cur_weight = planner_weighted_sum_module_vector[planner_weighted_sum_module_vector_name_dict[module_name]].vector
            # cur_weight = planner_weighted_sum_module_vector[planner_weighted_sum_module_vector_name_dict[module_name]].apply_to(planner_state_dict[m])
            for name, param in self.planner_new.named_parameters():
                # breakpoint()
                if name in detail_module_name:
                    name_number = detail_module_name.index(name)
                    # self.load_weights(self.planner_new, name, cur_weight[name_number])

                    
                    if len(name.split(".")) == 2:
                        param = getattr(getattr(self.planner_new, name.split(".")[0]), name.split(".")[1])  # model.linear.weight에 접근
                    elif len(name.split(".")) == 3:
                        param = getattr(getattr(getattr(self.planner_new, name.split(".")[0]), name.split(".")[1]), name.split(".")[2])

                    param.data += cur_weight[name_number]
                    
                    
        
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = batch_data["neg_seeds"], batch_data["neg_seeds_mask"], batch_data["neg_hist"], batch_data["neg_hist_mask"], batch_data["agent_mask"], batch_data["pad_mask_person"], batch_data["pos_seeds"], batch_data["pos_seeds_mask"], batch_data["pos_hist"], batch_data["pos_hist_mask"] 
        neg_seeds, neg_seeds_mask, neg_hist, neg_hist_mask, agent_mask, pad_mask_person, pos_seeds, pos_seeds_mask, pos_hist, pos_hist_mask = neg_seeds.cuda(), neg_seeds_mask.cuda(), neg_hist.cuda(), neg_hist_mask.cuda(), agent_mask.cuda(), pad_mask_person.cuda(), pos_seeds.cuda(), pos_seeds_mask.cuda(), pos_hist.cuda(), pos_hist_mask.cuda()  

        human_history = neg_hist   # B, N, H_T, 2
        human_states = neg_hist[:,:,-1,:]  # B, N, 2
        robot_states = pos_hist[:,-1,:]  # B, 2
        robot_states = torch.cat([robot_states, pos_hist[:, -1] - pos_hist[:, -2]], dim = -1) # B, 4
        
        # NOTE: Merging에서는 forecaster에도 loss를 흘려줌
        # vel_forecasts, _ = forecaster(human_history, pad_mask_person, neg_hist_mask)
        # forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
    
        if is_train:
            vel_forecasts, _ = self.forecaster_new(human_history, pad_mask_person, neg_hist_mask)
            forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
            vel_plan = self.planner_new(robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=forecasts)     
        else:
            with torch.no_grad():
                vel_forecasts, _ = self.forecaster_new(human_history, pad_mask_person, neg_hist_mask)
                forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
                vel_plan = self.planner_new(robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=forecasts)  # TODO: 평가에서는 mask 사용 불가?.

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)     # plan을 생성
        if batch_idx <= 3:
            if viz_dir is not None: viz_trajectory(plan.unsqueeze(1), pos_seeds, pos_hist.unsqueeze(1), viz_dir, batch_idx, agent_idx_=0, epoch_number=epoch_number)
        
        # for name, param in self.trainable_weights.named_parameters():
        #     print(f"{name}: {param.data[0]}")
        [print(f"{name}: {param.data[0]}") for name, param in self.trainable_weights.named_parameters()]
        breakpoint()

        if is_train:
            reg_loss = self.criterion(plan*pos_seeds_mask, pos_seeds.squeeze(1))              # pos seeds와 비교해서 loss를 줌
            col_loss = col_weight*get_chomp_col_loss_ETH(plan, neg_seeds, neg_seeds_mask)
            loss = reg_loss + col_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.data.item(), reg_loss.item(), col_loss.item()
        else:
            # breakpoint()
            ade, fde, col_gt, jerk, miss_rate, jerkrate = get_return_total(neg_mask = neg_seeds_mask, plans_sampled=plan.unsqueeze(1), plans_gt=pos_seeds, human_y_gt=neg_seeds, plan_horizon=pos_seeds.shape[-2], human_y_pred=None, train_mode=False)
            return sum(ade), sum(fde), sum(col_gt), sum(jerk), sum(miss_rate), sum(jerkrate), robot_states.shape[0]

        
        '''
        3. trainable weight와 pretrained parameter를 기반으로 테스크 벡터를 만듬. 
        4. load_state_dict 함수 동일하게 사용해서 로드해줌.  -> 3,4 동작을 매 학습 배치마다 해줘야함.
        5. load한 모델을 기반으로 출력 나오도록 함.
        6. 출력된 결과를 기반으로 loss backward를 수행.
        
        return None
        '''
        
        

