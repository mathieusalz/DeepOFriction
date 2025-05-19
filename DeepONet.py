import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BranchNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, y):
        return self.net(y)

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.iter = 0

    def forward(self, x, y):
        # x: [batch, branch_input_dim]
        # y: [batch, trunk_input_dim]
        branch_out = self.branch_net(x)  # [batch, latent_dim]
        trunk_out = self.trunk_net(y)    # [batch, latent_dim]
        out = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)  # [batch, 1]
        return out

    def loss(self, x, y, target):
        pred = self.forward(x, y)
        return F.mse_loss(pred, target)
    
    def inference(self, velocity, time):
        self.eval()  # Set the model to evaluation mode

        # Ensure velocity is a tensor of shape [1, 250]
        if velocity.ndimension() == 1:
            velocity = velocity.unsqueeze(0)  # [1, 250]
        
        # Ensure time is a tensor of shape [T, 1]
        if time.ndimension() == 1:
            time = time.unsqueeze(1)  # [T, 1]
        
        # Perform the forward pass for all time steps
        with torch.no_grad():
            predictions = self.forward(velocity, time)  # [1, T, 1]
        
        return predictions.squeeze()  # [T, 1] for the friction coefficients
    
    def closure(self, optimizer, x, y, target):
        def _closure():
            optimizer.zero_grad()
            loss = self.loss(x, y, target)
            loss.backward()

            if self.iter % 500 == 0:
                print(f"Loss: {loss.item()}")
            self.iter += 1
            return loss
        return _closure

        