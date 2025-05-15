import os
import json
import pandas as pd
import numpy as np
import gymnasium as gym
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor



class Data_Class:

    # Internal Class
    class Trajectory_Class:
        def __init__(self, traj_series):
            self.traj_list = traj_series
            self.length = len(traj_series)

        def get_single_traj_json(self, index):
            return json.loads(self.traj_list[index])
        
        def get_single_traj_dict(self, index):
            # return json.loads(self.traj_list[index])
            return self.traj_list[index]

        def get_single_traj(self, index):
            try:
                traj = self.get_single_traj_json(index)
            except json.JSONDecodeError:
                traj = self.get_single_traj_dict(index)

            return traj

    def __init__(self, path):
        self.path = path

        # 原始数据 Original Data
        self.trajs_prefer_list = []
        self.trajs_reject_list = []

        # 处理数据 Processed Data
        self.traj_prefer_list_list_tensor = []
        self.traj_reject_list_list_tensor = []

        # 启动函数 Start Function
        self.load_data(path)
        self.convert(self.trajs_prefer_list, self.traj_prefer_list_list_tensor)   # data convert 数据转换
        self.convert(self.trajs_reject_list, self.traj_reject_list_list_tensor)
        print("Data loaded from: ", path)

    def load_data(self, path):
        data = pd.read_csv(path)

        self.trajs_prefer_list = Data_Class.Trajectory_Class(data['preferred'])   # list data 数据
        self.trajs_reject_list = Data_Class.Trajectory_Class(data['rejected'])    # list data 数据

    def convert(self,
                LIST: Trajectory_Class,
                traj_list_list_tensor):

        # 获取第0条轨迹的第0时刻样本来确定维度
        # Get the first sample of the first trajectory to determine dimensions
        sample  = LIST.get_single_traj(0)[0]
        state0  = np.array(sample['state'])
        action0 = np.array(sample['action'])

        # 获取 state action 维度
        # Get the dimensions of state and action
        self.dim_state  = state0.size if state0.ndim == 0 else state0.shape[0]
        self.dim_action = action0.size if action0.ndim == 0 else action0.shape[0]

        # 数据批量转换 tensor
        # Convert data to tensor in batches
        for idx in range(LIST.length):
            traj = LIST.get_single_traj(idx)
            states, actions = [], []

            for time_i in traj:
                # 转换为 numpy，然后 torch tensor
                # Convert to numpy, then torch tensor
                state_np = np.array(time_i['state'])
                action_np = np.array(time_i['action'])

                state_t = torch.from_numpy(state_np).float()
                action_t = torch.from_numpy(action_np).float()

                # 如果是一维标量，要展开成长度1向量
                # If it's a one-dimensional scalar, expand it into a length 1 vector
                state_t = state_t.view(-1)
                action_t = action_t.view(-1)

                states.append(state_t)
                actions.append(action_t)

            # 将列表堆成张量 [L_i, dim]
            # Stack the list into a tensor [L_i, dim]
            states_tensor = torch.stack(states, dim=0)
            actions_tensor = torch.stack(actions, dim=0)

            # 将每条轨迹作为一个元组 (states, actions) 添加到列表中
            # Add each trajectory as a tuple (states, actions) to the list
            traj_list_list_tensor.append((states_tensor, actions_tensor))

# ——— 数据集与加载器 ———
# Dataset and DataLoader
class PreferenceDataset(Dataset):
    def __init__(self, pref, rej, gamma):
        assert len(pref) == len(rej)
        self.pref = pref
        self.rej = rej
        self.gamma = gamma

    def __len__(self):
        return len(self.pref)

    def __getitem__(self, idx):
        return (*self.pref[idx], *self.rej[idx])

# MLP 打分类 
# MLP scoring model class
class RewardMLP(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            #0000FF # 这里在构造神经网络，后去可能需要调整神经网络结构 
            #0000FF # Here we construct the neural network, which may need to be adjusted later 
            nn.Linear(s_dim + a_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, a):
        # s: [L_i, s_dim], a: [L_i, a_dim]
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)

