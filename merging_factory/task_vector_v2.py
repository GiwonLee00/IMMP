import torch
import torch.nn as nn
# NOTE: 기존 TaskVector는 weight의 저장 경로로 받아옴. 여기서는 weight를 바로 받아올 수 있도록 수정함.
class TaskVector(nn.Module):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, vector_module_name=None, vector_dataset_name=None, vector_epoch_name=None):
        super(TaskVector, self).__init__()
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            # with torch.no_grad():
                # pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                # finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
            pretrained_state_dict = pretrained_checkpoint
            finetuned_state_dict = finetuned_checkpoint
            # breakpoint()
            self.vector = []
            self.vector_module_name = vector_module_name
            self.vector_dataset_name = vector_dataset_name
            self.vector_epoch_name = vector_epoch_name
            for key in range(len(pretrained_state_dict)):
                # print("pretrained_state_dict[key]: ", pretrained_state_dict[key])
                if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    continue
                self.vector.append(finetuned_state_dict[key] - pretrained_state_dict[key])
            # breakpoint()
    
    def __add__(self, other):
        """Add two task vectors together."""
        # with torch.no_grad():
        new_vector = []
        for key in range(len(self.vector)):
            # breakpoint()
            '''
            if key not in other.vector:
                print(f'Warning, key {key} is not present in both task vectors.')
                continue
            '''
            new_vector.append(self.vector[key] + other.vector[key])
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        new_vector = []
        for key in range(len(self.vector)):
            new_vector.append(- self.vector[key])
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""    
        # NOTE: 250124, 내가 수정함
        # with torch.no_grad():
        # pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = []
        # pretrained_state_dict = pretrained_model.state_dict()
        pretrained_state_dict = pretrained_checkpoint
        # breakpoint()
        for key in range(len(pretrained_state_dict)):
            '''
            if key not in self.vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            '''
            new_state_dict.append(pretrained_state_dict[key] + scaling_coef * self.vector[key])
        # pretrained_model.load_state_dict(new_state_dict, strict=False)
        
        return new_state_dict


    # NOTE: 250124 기원 추가. Weight sum을 위해서 사용.
    def __mul__(self, factor):
        """Multiply a task vector by a scalar."""
        # with torch.no_grad():
        new_vector = []
        for key in range(len(self.vector)):
            # breakpoint()
            new_vector.append(self.vector[key] * factor)            # 기존 weight는 업데이트 하지 않음.
        return TaskVector(vector=new_vector)


