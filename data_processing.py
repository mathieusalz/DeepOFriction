import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DeepONetDataset(Dataset):
    def __init__(self, velocity, time, fric_coef):
        # Convert to tensors
        self.velocity = torch.tensor(velocity, dtype=torch.float32)  # [N_exp, 250]
        self.time = torch.tensor(time, dtype=torch.float32).unsqueeze(1)    # [T, 1]
        self.fric_coef = torch.tensor(fric_coef, dtype=torch.float32)  # [N_exp, 250]

        self.N, self.T = self.fric_coef.shape

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        exp_idx = idx // self.T
        time_idx = idx % self.T

        velocity_sample = self.velocity[exp_idx]           # [250]
        time_sample = self.time[time_idx]                  # [1]
        fric_value = self.fric_coef[exp_idx, time_idx]     # scalar

        return velocity_sample, time_sample, fric_value.unsqueeze(0)  # [250], [1], [1]


def get_data():

    velocity = pd.read_csv('synthetic_data_generation/features.csv', header = None)
    fric_coef = pd.read_csv('synthetic_data_generation/targets_AgingLaw.csv', header = None)

    max_fric = fric_coef.to_numpy().max()
    min_fric = fric_coef.to_numpy().min()

    max_vel = velocity.to_numpy().max()

    fric_coef = (np.array(fric_coef) - min_fric) / (max_fric - min_fric)

    velocity = np.log(np.log(1/np.array(velocity)))
    min_vel = velocity.min()
    max_vel = velocity.max()
    velocity = (velocity - min_vel) / (max_vel - min_vel)

    Vmax = 1.0e-1
    Dc = 1.0e-1
    nTransient = 300
    delta_t = (Dc / Vmax) / nTransient

    total_time = delta_t * 250
    time = np.arange(0, total_time, delta_t)

    return velocity, fric_coef, time