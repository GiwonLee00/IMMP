import torch
import torch.nn as nn

class ForecastNetworkAttention_ETH(nn.Module):
    """ Policy network for imitation learning """
    def __init__(self, embedding_dim=64, hidden_dim=64, local_dim=32, history=8, horizon=12):
        super().__init__()
        self.horizon = horizon
        self.history = history
        self.hidden_dim = hidden_dim
        self.history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        self.num_mode = 6

        self.human_encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.human_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.human_query = nn.Linear(hidden_dim, hidden_dim)
        self.human_key = nn.Linear(hidden_dim, hidden_dim)
        self.human_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.num_mode_expander = nn.Linear(hidden_dim*2, hidden_dim*self.num_mode)              # 128, 64*6
        self.human_forecaster = nn.Linear(hidden_dim, horizon*2)
    
    def forward(self, crowd_obsv, pad_mask):
        breakpoint()
        num_human = crowd_obsv.shape[2]
        crowd_obsv = crowd_obsv.permute(0,2,1,3) # TODO: 여기 맞느지 체크 
        crowd_obsv_ = crowd_obsv.reshape(-1, self.history+1, 2)
        _, (hn, _) = self.history_encoder(crowd_obsv_)
        history_encoded = hn[0].reshape(-1, num_human, self.hidden_dim)
        hidden = self.human_head(self.human_encoder(history_encoded))
        query = self.human_query(hidden)
        key = self.human_key(hidden)
        value = self.human_value(hidden)

        logits = torch.matmul(query, key.permute([0, 2, 1]))
        logits[~pad_mask] = -999999999999
        softmax = nn.functional.softmax(logits, dim=2)
        assert softmax[0,0,-1] == 0.0
        
        # NOTE: 내가 추가한 부분 ###################################################################################
        # is_uniform = torch.all(logits == logits[..., :1], dim=2, keepdim=True).squeeze(-1)
        # mask = torch.ones_like(softmax)
        # mask[is_uniform] = 0.0
        # softmax = softmax * mask
        ###########################################################################################################
        
        human_attentions = torch.matmul(softmax, value)
        
        summation = torch.cat([value, human_attentions], dim = -1).reshape(value.shape[0]*num_human, self.hidden_dim*2)    # 128*58, 128
        final_feature = self.num_mode_expander(summation).reshape(value.shape[0]*num_human, self.num_mode, self.hidden_dim)   # 128*58, 64*6 -> reshape (... , 6, 64)
        
        forecast = self.human_forecaster(final_feature)  # 128*58, 6, 8*2
        forecast = forecast.reshape(-1, num_human, self.num_mode, self.horizon, 2)
        return forecast, torch.cat([value, human_attentions], dim=-1)
        
        

