import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_solution(model, time, velocity, fric_coef, device):

    fig, axs = plt.subplots(5, 4, figsize=(20, 15))  # 5 rows, 4 columns

    for idx in range(20):
        row = idx // 4
        col = idx % 4

        ax1 = axs[row, col]
        
        # Plot velocity on primary y-axis
        ax1.plot(time, velocity[idx, :], label='Velocity', color='tab:blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Velocity', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_yscale('log')

        # Twin axis for friction
        ax2 = ax1.twinx()
        ax2.plot(time, fric_coef[idx, :], label='Friction Coefficient', color='tab:red')
        output = model.inference(torch.Tensor(velocity[idx, :]).to(device), torch.Tensor(time).to(device))
        ax2.plot(time, output, label='Predicted Friction Coefficient', color='tab:orange')
        ax2.set_ylabel('Friction', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax1.set_title(f"Sample {idx}")

    # Adjust layout and show
    plt.tight_layout()
    plt.savefig('results.png')
