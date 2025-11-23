import torch
import torch.nn as nn


class TaskVector(nn.Module):
    def __init__(self, pretrained=None, finetuned=None, module_l=None, vector=None, multimodule_l=None):
        super(TaskVector, self).__init__()
        if vector is not None:
            self.vector = vector
        elif finetuned is None:
            assert pretrained is not None
            self.vector = {}
            if multimodule_l is not None:       # NOTE: For Second Contribution
                for key, fake_key in zip(multimodule_l, module_l):
                    self.vector[key] = pretrained[fake_key]
            else:                               # 일반적인 케이스
                for key in module_l:
                    self.vector[key] = pretrained[key]
        else:
            assert pretrained is not None
            self.vector = {}
            if multimodule_l is not None:       # NOTE: For Second Contribution
                for key, fake_key in zip(multimodule_l, module_l):
                    self.vector[key] = finetuned[fake_key] - pretrained[fake_key]
            else:
                for key in module_l:
                    self.vector[key] = finetuned[key] - pretrained[key]
        self.module_l = module_l
    
    def __add__(self, other):
        new_vector = {}
        for key in self.vector:
            if key not in other.vector:
                print(f'Warning, key {key} is not present in both task vectors.')
                continue
            new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        new_vector = {}
        for key in self.vector:
            new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        new_state_dict = {}
        pretrained_state_dict = pretrained_checkpoint
        for key in pretrained_state_dict:
            if key not in self.vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        return new_state_dict

    def __mul__(self, factor):
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * factor
            return TaskVector(vector=new_vector)

