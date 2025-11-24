# TODO: 250209에 옮김. 변경사항 생기면 다시 수정.
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class PlannerModel_uni(nn.Module):
    """ Policy network for imitation learning """
    def __init__(self, hidden_dim=64, horizon=12, history = 8):
        super().__init__()
        self.horizon = horizon
        self.history = history
        self.hidden_dim = hidden_dim

        self.robot_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        self.forecast_encoder = nn.LSTM(2, hidden_dim, batch_first=True)

        # ------------------------------------------------------ # 
        self.robot_history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)  # TODO 추가해둠. 비교 후에 삭제 ㄱㄱ
        self.robot_encoder_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # ------------------------------------------------------ # 

        self.human_encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.robot_query = nn.Linear(hidden_dim, hidden_dim)
        self.human_key = nn.Linear(hidden_dim, hidden_dim)
        self.human_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.task_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )  
        
        self.robot_plan = nn.Linear(hidden_dim*2, horizon*2)
    
    def forward(self, robot_state, crowd_obsv, pad_mask, neg_hist_mask, pos_hist, pos_hist_mask, forecasts=None):   # crowd_obsv = human_history!!
        batch_size, num_human, seq_len, _ = crowd_obsv.shape
        crowd_obsv = crowd_obsv * neg_hist_mask     # invalid part는 0처리
        crowd_obsv_ = crowd_obsv.reshape(-1, self.history, 2)

        # packing crowd_obsv (lstm)
        zero_mask = (crowd_obsv_ == 0)
        last_zero_idx = (zero_mask * torch.arange(crowd_obsv_.size(1), device=crowd_obsv_.device).unsqueeze(-1)).amax(dim=1)
        last_zero_idx[~zero_mask.any(dim=1)] = -1  # 0이 아예 없는 경우 -1로 설정
        last_zero_idx = last_zero_idx[:, 0]
        row_indices = torch.arange(crowd_obsv_.size(1), device=crowd_obsv_.device).expand(crowd_obsv_.size(0), -1)  # (4800, 8) 크기의 인덱스 행렬
        mask = (row_indices < last_zero_idx.unsqueeze(1)) & (last_zero_idx.unsqueeze(1) >= 0)  # 마지막 0 이전만 True
        crowd_obsv_[mask] = 0
        real_time_length = (7 - last_zero_idx).detach().cpu()
        crowd_obsv_ = rnn_utils.pack_padded_sequence(crowd_obsv_, real_time_length.clamp(min=1), batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.history_encoder(crowd_obsv_)
        history_encoded = hn[0].reshape(-1, num_human, self.hidden_dim)
        
        # forecasts는 뭐가 실제 경로인 지 모름, 그냥 생성된 거니까 그대로 넣어줄 수 있음.
        forecasts = forecasts.reshape(-1, self.horizon, 2)
        _, (fn, _) = self.forecast_encoder(forecasts)
        future_encoded = fn[0].reshape(-1, num_human, self.hidden_dim)
        human_obs = torch.cat([history_encoded, future_encoded], dim = -1)
        human_forecasts_emb = self.human_encoder(human_obs)

        
        pos_hist = pos_hist * pos_hist_mask
        pos_hist_ = pos_hist.reshape(-1, self.history, 2)

        # packing pos_hist (lstm)   # SIT에서 invalid한 로봇 경로 존재. 마스킹 해줌.
        robot_zero_mask = (pos_hist_ == 0)
        robot_last_zero_idx = (robot_zero_mask * torch.arange(pos_hist_.size(1), device=pos_hist_.device).unsqueeze(-1)).amax(dim=1)
        robot_last_zero_idx[~robot_zero_mask.any(dim=1)] = -1  # 0이 아예 없는 경우 -1로 설정
        robot_last_zero_idx = robot_last_zero_idx[:, 0]
        robot_row_indices = torch.arange(pos_hist_.size(1), device=pos_hist_.device).expand(pos_hist_.size(0), -1)  # (4800, 8) 크기의 인덱스 행렬
        robot_mask = (robot_row_indices < robot_last_zero_idx.unsqueeze(1)) & (robot_last_zero_idx.unsqueeze(1) >= 0)  # 마지막 0 이전만 True
        pos_hist_[robot_mask] = 0
        robot_real_time_length = (7 - robot_last_zero_idx).detach().cpu()
        pos_hist_ = rnn_utils.pack_padded_sequence(pos_hist_, robot_real_time_length.clamp(min=1), batch_first=True, enforce_sorted=False)    # 나머지 애들은 어차피 invalid하게 날라감. 일단 1로 채워줌

        _, (rhn, _) = self.robot_history_encoder(pos_hist_)
        robot_his = rhn[0].reshape(-1,self.hidden_dim)
        robot_cur = self.robot_encoder(robot_state[:, :4])
        robot_cats = torch.cat([robot_his, robot_cur], dim = -1)
        robot_emb = self.robot_encoder_2(robot_cats)

        query = self.robot_query(robot_emb)
        key = self.human_key(human_forecasts_emb)
        value = self.human_value(human_forecasts_emb)

        logits = torch.matmul(query.view(-1, 1, self.hidden_dim), key.permute([0, 2, 1]))
        logits[~(pad_mask[:,0,:].unsqueeze(1))] = -999999999999
        softmax = nn.functional.softmax(logits, dim=2)
        # assert softmax[0,0,-1] == 0.0     # crowdnav는 모든 human이 valid
        
        human_attentions = torch.matmul(softmax, value)
        reparam_robot_state = torch.cat([robot_state[:, -2:] - robot_state[:, :2], robot_state[:, 2:4]], axis=1)
        robot_task = self.task_encoder(reparam_robot_state)
        
        plan = self.robot_plan(torch.cat([robot_task, human_attentions.squeeze(1)], dim = -1))
        plan = plan.view(-1, self.horizon, 2)
        
        return plan
        
        

