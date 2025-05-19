import torch
from torch.utils.data import DataLoader
from DeepONet import DeepONet, BranchNet, TrunkNet
from data_processing import DeepONetDataset, get_data
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from time import time
from postprocessing import plot_solution

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(device)}", flush=True)
        #torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark=True
    else:
        print("Using device: CPU", flush=True)

    velocity, fric_coef, time_arr = get_data()
    
    dataset = DeepONetDataset(velocity, time_arr, fric_coef)
    
    # Need to test different amount of workers + pin memory
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

    bNet = BranchNet(input_dim=velocity.shape[1], output_dim=32).to(device)
    tNet = TrunkNet(input_dim=1, output_dim=32).to(device)
    model = DeepONet(bNet, tNet).to(device)
    # model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #scaler = torch.cuda.amp.GradScaler()

    start = time()
    for epoch in range(500):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            vel_batch, time_batch, target_batch = batch
            vel_batch = vel_batch.to(device, non_blocking=True)
            time_batch = time_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            # closure_fn = model.closure(optimizer_BFGS, vel_batch, time_batch, target_batch)
            # optimizer_BFGS.step(closure_fn)
            # loss = optimizer_BFGS.step(model.closure(optimizer_BFGS, vel_batch, time_batch, target_batch))
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            loss = model.loss(vel_batch, time_batch, target_batch)

            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            running_loss += loss.item() * vel_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.3e}", flush=True)

    training_time = time() - start
    print(f"Total Training time: {training_time:.2f} seconds \t Time per epoch: {training_time/500:.2f}", flush=True)

    plot_solution(model, time_arr, velocity, fric_coef, device)
    torch.save(model.to('cpu').state_dict(), 'model.pth')
    print("Model saved to model.pth", flush=True)


if __name__ == "__main__":
    train()