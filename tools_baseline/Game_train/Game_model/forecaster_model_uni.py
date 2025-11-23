# TODO: 250209에 옮김. 변경사항 생기면 다시 수정.
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class ForecasterModel_uni(nn.Module):
    """ Policy network for imitation learning """
    def __init__(self, embedding_dim=64, hidden_dim=64, local_dim=32, history=8, horizon=12):
        super().__init__()
        self.horizon = horizon
        self.history = history
        self.hidden_dim = hidden_dim
        
        self.history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)   
        
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
        
        self.human_forecaster = nn.Linear(hidden_dim*2, horizon*2)
    
    def forward(self, crowd_obsv, pad_mask, neg_hist_mask):
        batch_size, num_human, seq_len, _ = crowd_obsv.shape
        crowd_obsv = crowd_obsv * neg_hist_mask     # invalid part는 0처리
        crowd_obsv_ = crowd_obsv.reshape(-1, self.history, 2)   # 각 경로들을 따로 따로 transformer 로 임베딩.

        zero_mask = (crowd_obsv_ == 0)
        last_zero_idx = (zero_mask * torch.arange(crowd_obsv_.size(1), device=crowd_obsv_.device).unsqueeze(-1)).amax(dim=1)
        last_zero_idx[~zero_mask.any(dim=1)] = -1  # 0이 아예 없는 경우 -1로 설정
        last_zero_idx = last_zero_idx[:, 0]
        row_indices = torch.arange(crowd_obsv_.size(1), device=crowd_obsv_.device).expand(crowd_obsv_.size(0), -1)  # (4800, 8) 크기의 인덱스 행렬
        mask = (row_indices < last_zero_idx.unsqueeze(1)) & (last_zero_idx.unsqueeze(1) >= 0)  # 마지막 0 이전만 True
        crowd_obsv_[mask] = 0
        real_time_length = (7 - last_zero_idx).detach().cpu()
        crowd_obsv_ = rnn_utils.pack_padded_sequence(crowd_obsv_, real_time_length.clamp(min=1), batch_first=True, enforce_sorted=False)    # 나머지 애들은 어차피 invalid하게 날라감. 일단 1로 채워줌
        
        _, (hn, _) = self.history_encoder(crowd_obsv_)
        history_encoded = hn[0].reshape(-1, num_human, self.hidden_dim)

        hidden = self.human_head(self.human_encoder(history_encoded))
        query = self.human_query(hidden)
        key = self.human_key(hidden)
        value = self.human_value(hidden)
        logits = torch.matmul(query, key.permute([0, 2, 1]))
        logits[~pad_mask] = -999999999999
        softmax = nn.functional.softmax(logits, dim=2)
        human_attentions = torch.matmul(softmax, value)
        
        forecast = self.human_forecaster(torch.cat([value, human_attentions], dim = -1))
        forecast = forecast.reshape(-1, num_human, self.horizon, 2)
        
        return forecast, torch.cat([value, human_attentions], dim=-1)

        
        

