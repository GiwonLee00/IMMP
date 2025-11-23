import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from policy.forecaster_model_uni import ForecasterModel_uni
from policy.planner_model_uni import PlannerModel_uni     

from merging_factory.task_vector_woo import TaskVector
from merging_factory.weight_extracter_woo import div_forecaster, div_planner, div_foreplanner

import torch.optim as optim
import pytorch_lightning as pl
import pathlib
from utils.viz_utils import *
from utils.metric import *  

from collections import defaultdict
from functools import reduce


def nested_dict():
    return defaultdict(nested_dict)


def Collect_Vector(cfg, args, forecaster_sample, planner_sample):
    dataset = cfg["total_dataset_list"][0]
    forecaster_0 = torch.load(f"{args.pretrained_path}/forecaster/{dataset}/ckpt/initial_weight.pth", weights_only=True)
    planner_0 = torch.load(f"{args.pretrained_path}/planner/{dataset}{args.planner_exp_name}/ckpt/initial_weight.pth", weights_only=True)        
    
    forecaster_n_list, forecaster_m_list = div_forecaster(forecaster_0.keys())
    planner_n_list, planner_m_list = div_planner(planner_0.keys())
    
    init_vector_f={}; init_vector_p={}
    for name, module_l in zip(forecaster_n_list, forecaster_m_list):
        init_vector_f[name]= TaskVector(pretrained=forecaster_0, module_l=module_l)

    for name, module_l in zip(planner_n_list, planner_m_list):
        init_vector_p[name] = TaskVector(pretrained=planner_0, module_l=module_l)

    forecaster_tvd, planner_tvd = nested_dict(), nested_dict()
    for dataset in cfg["total_dataset_list"]:
        # breakpoint()
        # if dataset == args.test_dataset_name: continue
        if dataset not in args.except_dataset_name: continue
        ##################
        for sample in forecaster_sample:   
            forecaster_T = torch.load(f"{args.pretrained_path}/forecaster/{dataset}/ckpt/{sample}.pth", weights_only=True)   
            for name, module_l in zip(forecaster_n_list, forecaster_m_list):  
                vector = TaskVector(forecaster_0, forecaster_T, module_l)
                forecaster_tvd[dataset][sample][name] = vector
        ##################    
        for sample in planner_sample:
            planner_T = torch.load(f"{args.pretrained_path}/planner/{dataset}{args.planner_exp_name}/ckpt/{sample}.pth", weights_only=True)
            for name, module_l in zip(planner_n_list, planner_m_list):
                # breakpoint()
                vector = TaskVector(planner_0, planner_T, module_l)
                planner_tvd[dataset][sample][name] = vector

    return init_vector_f, init_vector_p, forecaster_tvd, planner_tvd, [forecaster_n_list, forecaster_m_list], [planner_n_list, planner_m_list]

def Collect_Vector_foreplan(cfg, args, forecaster_sample, planner_sample):
    dataset = cfg["total_dataset_list"][0]
    forecaster_0 = torch.load(f"{args.pretrained_path}/forecaster/{dataset}/ckpt/initial_weight.pth", weights_only=True)
    planner_0 = torch.load(f"{args.pretrained_path}/planner/{dataset}{args.planner_exp_name}/ckpt/initial_weight.pth", weights_only=True)        
    
    forecaster_n_list, forecaster_m_list = div_forecaster(forecaster_0.keys())
    planner_n_list, planner_m_list = div_planner(planner_0.keys())
    foreplan_n_list, foreplan_m_list, planfore_n_list, planfore_m_list = div_foreplanner(forecaster_0.keys(), planner_0.keys())
    
    init_vector_f={}; init_vector_p={}
    for name, module_l in zip(forecaster_n_list, forecaster_m_list):
        init_vector_f[name]= TaskVector(pretrained=forecaster_0, module_l=module_l)

    for name, module_l in zip(planner_n_list, planner_m_list):
        init_vector_p[name] = TaskVector(pretrained=planner_0, module_l=module_l)

    forecaster_tvd, planner_tvd = nested_dict(), nested_dict()
    for dataset in cfg["total_dataset_list"]:
        # breakpoint()
        # if dataset == args.test_dataset_name: continue
        if dataset not in args.except_dataset_name: continue
        ##################
        for sample in forecaster_sample:   
            forecaster_T = torch.load(f"{args.pretrained_path}/forecaster/{dataset}/ckpt/{sample}.pth", weights_only=True)   
            for name, module_l in zip(forecaster_n_list, forecaster_m_list):
                vector = TaskVector(forecaster_0, forecaster_T, module_l)
                forecaster_tvd[dataset][sample][name] = vector
        ##################    
        for sample in planner_sample:
            planner_T = torch.load(f"{args.pretrained_path}/planner/{dataset}{args.planner_exp_name}/ckpt/{sample}.pth", weights_only=True)
            for name, module_l in zip(planner_n_list, planner_m_list):
                vector = TaskVector(planner_0, planner_T, module_l)
                planner_tvd[dataset][sample][name] = vector

        # NOTE: Second Contribution; learning with plan vector --------------------------------------- # 
        for sample in planner_sample:           # planner sample에 있는거로 사용해서 넣어줌.
            planner_T = torch.load(f"{args.pretrained_path}/planner/{dataset}{args.planner_exp_name}/ckpt/{sample}.pth", weights_only=True)
            for name, module_l in zip(planner_n_list, planner_m_list):
                # breakpoint()
                for name, multimodule_l, plan_name, plan_multimodule_l in zip(foreplan_n_list, foreplan_m_list, planfore_n_list, planfore_m_list):
                    if module_l != plan_multimodule_l: continue
                    # breakpoint()
                    vector = TaskVector(planner_0, planner_T, module_l, vector=None, multimodule_l=multimodule_l)
                    
                    new_sample = sample + "_foreplan"
                    forecaster_tvd[dataset][new_sample][name] = vector

        # breakpoint()  
    return init_vector_f, init_vector_p, forecaster_tvd, planner_tvd, [forecaster_n_list, forecaster_m_list], [planner_n_list, planner_m_list]

def set_requires_grad(d):
    if isinstance(d, torch.Tensor):
        d.requires_grad = True
    elif isinstance(d, dict):
        for key, value in d.items():
            set_requires_grad(value)
    elif isinstance(d, defaultdict):
        for key, value in d.items():
            set_requires_grad(value)
    elif isinstance(d, TaskVector):
        set_requires_grad(d.vector)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class Merged_Model(nn.Module):
    def __init__(self, _forecaster, _planner, f_prior=0.3, p_prior=0.3):
        super().__init__()
        make_functional(_forecaster[0]); make_functional(_planner[0])
        self.forecaster, self.planner = _forecaster[0], _planner[0]

        self.init_vector_f, self.init_vector_p = _forecaster[1], _planner[1]
        set_requires_grad(self.init_vector_f); set_requires_grad(self.init_vector_p)
        
        self.forecaster_tvd, self.planner_tvd = _forecaster[2], _planner[2]
        set_requires_grad(self.forecaster_tvd); set_requires_grad(self.planner_tvd)

        self.f_spec, self.p_spec = _forecaster[3], _planner[3]
        self.f_list, self.p_list = _forecaster[4], _planner[4]

        forecaster_rlambdas = torch.ones(self.f_spec) * f_prior
        planner_rlambdas = torch.ones(self.p_spec) * p_prior
        self.forecaster_lambdas = torch.nn.Parameter(forecaster_rlambdas)
        self.planner_lambdas = torch.nn.Parameter(planner_rlambdas)


    def collect_trainable_params(self):
        return [self.forecaster_lambdas, self.planner_lambdas]


    def merging_weights(self):
        vector_sum_f = {name:0 for name in self.f_list[0]}
        for idx_d, (dataset, m_dict) in enumerate(self.forecaster_tvd.items()):
            for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                for idx_m, (module, vector) in enumerate(n_dict.items()):
                    vector_sum_f[module] += vector*self.forecaster_lambdas[idx_d][idx_e][idx_m]
                
        for module in vector_sum_f.keys():
            vector_sum_f[module] += self.init_vector_f[module]
            vector_sum_f[module].module_l = self.init_vector_f[module].module_l

        ##############
        vector_sum_p = {name:0 for name in self.p_list[0]}
        for idx_d, (dataset, m_dict) in enumerate(self.planner_tvd.items()):
            for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                for idx_m, (module, vector) in enumerate(n_dict.items()):
                    vector_sum_p[module] += vector*self.planner_lambdas[idx_d][idx_e][idx_m]

        for module in vector_sum_p.keys():
            vector_sum_p[module] += self.init_vector_p[module]
            vector_sum_p[module].module_l = self.init_vector_p[module].module_l

        ##############
        for module in vector_sum_f.keys():
            names = vector_sum_f[module].vector.keys()
            params = tuple(p for p in vector_sum_f[module].vector.values())
            load_weights(self.forecaster, names, params)

        for module in vector_sum_p.keys():
            names = vector_sum_p[module].vector.keys()
            params = tuple(p for p in vector_sum_p[module].vector.values())
            load_weights(self.planner, names, params)


    def save_weights(self, dir, type=None):
        state_dict = {}
        
        if type=='forecaster':
            vector_sum_f = {name:0 for name in self.f_list[0]}
            for idx_d, (dataset, m_dict) in enumerate(self.forecaster_tvd.items()):
                for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                    for idx_m, (module, vector) in enumerate(n_dict.items()):
                        vector_sum_f[module] += vector*self.forecaster_lambdas[idx_d][idx_e][idx_m]
                
            for module in vector_sum_f.keys():
                vector_sum_f[module] += self.init_vector_f[module]
                vector_sum_f[module].module_l = self.init_vector_f[module].module_l
        
            for module in vector_sum_f.keys():
                names = vector_sum_f[module].vector.keys()
                params = tuple(p for p in vector_sum_f[module].vector.values())
                state_dict.update(dict(zip(names, params)))

        
        elif type=='planner':
            vector_sum_p = {name:0 for name in self.p_list[0]}
            for idx_d, (dataset, m_dict) in enumerate(self.planner_tvd.items()):
                for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                    for idx_m, (module, vector) in enumerate(n_dict.items()):
                        vector_sum_p[module] += vector*self.planner_lambdas[idx_d][idx_e][idx_m]

            for module in vector_sum_p.keys():
                vector_sum_p[module] += self.init_vector_p[module]
                vector_sum_p[module].module_l = self.init_vector_p[module].module_l
                
            for module in vector_sum_p.keys():
                names = vector_sum_p[module].vector.keys()
                params = tuple(p for p in vector_sum_p[module].vector.values())
                state_dict.update(dict(zip(names, params)))
                
        torch.save(state_dict, dir)

    def save_weights_with_parameter(self, dir,dir2, type=None):
        state_dict = {}
        
        if type=='forecaster':
            vector_sum_f = {name:0 for name in self.f_list[0]}
            for idx_d, (dataset, m_dict) in enumerate(self.forecaster_tvd.items()):
                for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                    for idx_m, (module, vector) in enumerate(n_dict.items()):
                        vector_sum_f[module] += vector*self.forecaster_lambdas[idx_d][idx_e][idx_m]
                
            for module in vector_sum_f.keys():
                vector_sum_f[module] += self.init_vector_f[module]
                vector_sum_f[module].module_l = self.init_vector_f[module].module_l
        
            for module in vector_sum_f.keys():
                names = vector_sum_f[module].vector.keys()
                params = tuple(p for p in vector_sum_f[module].vector.values())
                state_dict.update(dict(zip(names, params)))

        
        elif type=='planner':
            vector_sum_p = {name:0 for name in self.p_list[0]}
            for idx_d, (dataset, m_dict) in enumerate(self.planner_tvd.items()):
                for idx_e, (epoch, n_dict) in enumerate(m_dict.items()):
                    for idx_m, (module, vector) in enumerate(n_dict.items()):
                        vector_sum_p[module] += vector*self.planner_lambdas[idx_d][idx_e][idx_m]

            for module in vector_sum_p.keys():
                vector_sum_p[module] += self.init_vector_p[module]
                vector_sum_p[module].module_l = self.init_vector_p[module].module_l
                
            for module in vector_sum_p.keys():
                names = vector_sum_p[module].vector.keys()
                params = tuple(p for p in vector_sum_p[module].vector.values())
                state_dict.update(dict(zip(names, params)))
                
        torch.save(state_dict, dir)
        if type=='forecaster':
            torch.save(self.forecaster_lambdas, dir2)
        elif type=='planner':
            torch.save(self.planner_lambdas, dir2)

    def forward(self, human_states, robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask):
        self.merging_weights()

        vel_forecasts, _ = self.forecaster(human_history, pad_mask_person, neg_hist_mask)
        forecasts = human_states[:, :, :2].unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = self.planner(robot_states, human_history, pad_mask_person, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=forecasts)     
        
        return vel_plan