import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
import ast
class DpoTrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        """
        pkl_path: path to the .pkl file, which is a list of dicts.
        Each item has:
        {
            "obs_pref": [T, obs_dim],
            "act_pref": [T, act_dim],
            "rew_pref": scalar,
            "obs_rej":  [T, obs_dim],
            "act_rej":  [T, act_dim],
            "rew_rej":  scalar,
        }
        """
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "obs_pref": torch.tensor(item["obs_pref"], dtype=torch.float32),  # [T, obs_dim]
            "act_pref": torch.tensor(item["act_pref"], dtype=torch.float32),  # [T, act_dim]
            "rew_pref": torch.tensor(item["rew_pref"], dtype=torch.float32),  # scalar
            "obs_rej":  torch.tensor(item["obs_rej"],  dtype=torch.float32),
            "act_rej":  torch.tensor(item["act_rej"],  dtype=torch.float32),
            "rew_rej":  torch.tensor(item["rew_rej"],  dtype=torch.float32)
        }

class PrefDataset(Dataset):
    def __init__(self, df):
        
        for col in ['initial_state', 'preferred', 'rejected']:
            df[col] = df[col].apply(ast.literal_eval)
        
        self.samples = []
        for _, row in df.iterrows():
            # Convert initial state to tensor
            state = torch.tensor(row['initial_state'], dtype=torch.float32)

            # Convert each step in trajectories to (state_tensor, action_tensor)
            traj_w = [(torch.tensor(step['state'], dtype=torch.float32),
                       torch.tensor(step['action'], dtype=torch.long)) 
                      for step in row['preferred']]

            traj_l = [(torch.tensor(step['state'], dtype=torch.float32),
                       torch.tensor(step['action'], dtype=torch.long)) 
                      for step in row['rejected']]

            self.samples.append((state, traj_w, traj_l))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def collate_fn(batch):
    # batch is a list of tuples: (initial_state_tensor, traj_w_tensor_list, traj_l_tensor_list)
    initial_states, traj_ws, traj_ls = zip(*batch)
    return list(initial_states), list(traj_ws), list(traj_ls)
