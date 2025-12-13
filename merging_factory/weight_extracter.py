import torch.nn as nn

def weight_extracter_forecaster(weight=None, keys_list=None):
    # (['history_encoder.weight_ih_l0', 'history_encoder.weight_hh_l0', 'history_encoder.bias_ih_l0', 'history_encoder.bias_hh_l0', 'human_encoder.1.weight', 'human_encoder.1.bias', 'human_encoder.3.weight', 'human_encoder.3.bias', 'human_head.0.weight', 'human_head.0.bias', 'human_query.weight', 'human_query.bias', 'human_key.weight', 'human_key.bias', 'human_value.weight', 'human_value.bias', 'human_forecaster.weight', 'human_forecaster.bias'])
    """
    # NOTE: LSTM part
    self.history_encoder

    # NOTE: Attention part
    self.human_query
    self.human_key
    self.human_value
    """
    
    LSTM_weight_keys = ['history_encoder.weight_ih_l0', 'history_encoder.weight_hh_l0', 'history_encoder.bias_ih_l0', 'history_encoder.bias_hh_l0']
    Attention_weight_keys = ['human_query.weight', 'human_query.bias', 'human_key.weight', 'human_key.bias', 'human_value.weight', 'human_value.bias']
    excluded_keys = LSTM_weight_keys + Attention_weight_keys
    # breakpoint()
    LSTM_list = [weight[keys_list.index(key)] for key in LSTM_weight_keys if key in keys_list]
    ATTN_list = [weight[keys_list.index(key)] for key in Attention_weight_keys if key in keys_list]
    ELSE_list = [weight[keys_list.index(key)] for key in keys_list if key not in excluded_keys]
    ELSE_key_list = [key for key in keys_list if key not in excluded_keys]

    # LSTM_dict = {key: weight[key] for key in LSTM_weight_keys if key in weight}
    # Attention_dict = {key: weight[key] for key in Attention_weight_keys if key in weight}
    # ELSE_dict = {key: weight[key] for key in weight if key not in excluded_keys}
    
    Total_list = [LSTM_list, ATTN_list, ELSE_list]     # 핵심 weight를 저장
    name_list = ["LSTM", "ATTN", "ELSE"]
    name_detail_list = [LSTM_weight_keys, Attention_weight_keys, ELSE_key_list]
    # breakpoint()

    return Total_list, name_list, name_detail_list

def weight_extracter_planner(weight=None, keys_list=None):
    # (['robot_encoder.0.weight', 'robot_encoder.0.bias', 'robot_encoder.2.weight', 'robot_encoder.2.bias', 'history_encoder.weight_ih_l0', 'history_encoder.weight_hh_l0', 'history_encoder.bias_ih_l0', 'history_encoder.bias_hh_l0', 'forecast_encoder.weight_ih_l0', 'forecast_encoder.weight_hh_l0', 'forecast_encoder.bias_ih_l0', 'forecast_encoder.bias_hh_l0', 'robot_history_encoder.weight_ih_l0', 'robot_history_encoder.weight_hh_l0', 'robot_history_encoder.bias_ih_l0', 'robot_history_encoder.bias_hh_l0', 'robot_encoder_2.1.weight', 'robot_encoder_2.1.bias', 'human_encoder.1.weight', 'human_encoder.1.bias', 'robot_query.weight', 'robot_query.bias', 'human_key.weight', 'human_key.bias', 'human_value.weight', 'human_value.bias', 'task_encoder.0.weight', 'task_encoder.0.bias', 'task_encoder.2.weight', 'task_encoder.2.bias', 'robot_plan.weight', 'robot_plan.bias'])
    """
    # NOTE: LSTM part
    self.history_encoder        # skip
    self.forecast_encoder       # skip
    self.robot_history_encoder

    # NOTE: Attention part
    self.robot_query 
    self.human_key
    self.human_value
    """

    LSTM_weight_keys = ['robot_history_encoder.weight_ih_l0', 'robot_history_encoder.weight_hh_l0', 'robot_history_encoder.bias_ih_l0', 'robot_history_encoder.bias_hh_l0']
    Attention_weight_keys = ['robot_query.weight', 'robot_query.bias', 'human_key.weight', 'human_key.bias', 'human_value.weight', 'human_value.bias']
    excluded_keys = LSTM_weight_keys + Attention_weight_keys

    LSTM_list = [weight[keys_list.index(key)] for key in LSTM_weight_keys if key in keys_list]
    ATTN_list = [weight[keys_list.index(key)] for key in Attention_weight_keys if key in keys_list]
    ELSE_list = [weight[keys_list.index(key)] for key in keys_list if key not in excluded_keys]
    ELSE_key_list = [key for key in keys_list if key not in excluded_keys]

    Total_list = [LSTM_list, ATTN_list, ELSE_list]     # 핵심 weight를 저장
    name_list = ["LSTM", "ATTN", "ELSE"]
    name_detail_list = [LSTM_weight_keys, Attention_weight_keys, ELSE_key_list]

    return Total_list, name_list, name_detail_list

