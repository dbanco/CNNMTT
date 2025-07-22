# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:52:09 2025

@author: dpqb1
"""
import torch

import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import argparse
from mtt_cnn import MTTModel
from synth_xray_data import MTTSyntheticDataset

import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

import wandb

def cleanup():
    dist.destroy_process_group()

def train(model, train_loader, criterion, optimizer, device_id, num_epochs, train_sampler):
    train_losses = []

    # Create output/log/checkpoint dirs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/run_{timestamp}'
    log_path = os.path.join(output_dir, 'train.log')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # comment out if running on SLURM where stdout is captured
        ]
    )
    
    logging.info(f"Starting new training run — outputs in {output_dir}")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, truths) in enumerate(train_loader):
            inputs, truths = inputs.to(device_id, non_blocking=True), truths.to(device_id, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, truths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if dist.get_rank() == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        # Save checkpoint
        if dist.get_rank() == 0:
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] average loss: {epoch_loss:.6f}")
            wandb.log({"epoch": epoch+1, "train_loss": epoch_loss})
            
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss
            }, checkpoint_path)
    
            logging.info(f"Saved checkpoint: {checkpoint_path}")

    return train_losses

def visualize_sequence(model, data_loader, device_id, save_dir, prefix="sequence", time_indices=[0,4,8,12,16,24,28]):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        inputs, truths = next(iter(data_loader))  # (T, 1, H, W)
        inputs, truths = inputs.to(device_id), truths.to(device_id)

        fig, axs = plt.subplots(len(time_indices), 3, figsize=(10, 2.5 * len(time_indices)))
        fig.suptitle(f"{prefix} Visualization", fontsize=14)

        for row_idx, t in enumerate(time_indices):
            input_frame = inputs[t].unsqueeze(0)   # (1,1,H,W)
            truth_frame = truths[t, 0].cpu().numpy()
            output_frame = model(input_frame).squeeze().cpu().numpy()

            axs[row_idx, 0].imshow(input_frame[0, 0].cpu().numpy(), cmap='gray')
            axs[row_idx, 0].set_title(f"Input t={t}")
            axs[row_idx, 1].imshow(truth_frame, cmap='viridis')
            axs[row_idx, 1].set_title(f"Truth t={t}")
            axs[row_idx, 2].imshow(output_frame, cmap='viridis')
            axs[row_idx, 2].set_title(f"Output t={t}")

            for col in range(3):
                axs[row_idx, col].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename = os.path.join(save_dir, f"{prefix}_visualization.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved visualization to {filename}")

# Shared global dataset
_shared_dataset = None

def get_shared_dataset(args):
    global _shared_dataset
    if _shared_dataset is None:
        _shared_dataset = MTTSyntheticDataset(
            num_spots=3,
            num_samples=args.num_train_samples,
            sequence_length=args.sequence_length,
            noise=True,
            input_shape=(1, args.height, args.width),
            seed=42,
            preload=True
        )
    return _shared_dataset

def main(args):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    g_rank = dist.get_rank()
    device_id = g_rank % torch.cuda.device_count()
    
    if g_rank == 0:
        print(f"[Rank {g_rank}] Preloading dataset...", flush=True)
        train_dataset = get_shared_dataset(args)
    dist.barrier()  # Wait for preload
    if g_rank != 0:
        train_dataset = get_shared_dataset(args)
    
    # Set seeds for reproducibility
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    model = MTTModel(input_channels=1, output_channels=args.num_outputs).to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[RANK {g_rank}] Model parameter count: {param_count}", flush=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,num_workers=16,pin_memory=True,)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if dist.get_rank() == 0:
        wandb.init(project="mtt-cnn", config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "architecture": "MTTModel"
        })
        wandb.watch(model, log="all")
    
    train(model, train_loader, criterion, optimizer, device_id, args.num_epochs, train_sampler)

    test_dataset = MTTSyntheticDataset(num_spots=3,
                                        num_samples=1,
                                        sequence_length=args.sequence_length,
                                        noise=True,
                                        input_shape=(1, args.height, args.width),
                                        seed=2,
                                        preload=True)
    test_loader = DataLoader(test_dataset, batch_size=args.sequence_length)
    train_loader_vis = DataLoader(train_dataset, batch_size=args.sequence_length, shuffle=False)

    if dist.get_rank() == 0:
        save_dir = "./visualizations"
        visualize_sequence(model, train_loader_vis, device_id, save_dir, prefix="train")
        visualize_sequence(model, test_loader, device_id, save_dir, prefix="test")
        
        wandb.log({
        "Train Sequence Visualization": wandb.Image(f"{save_dir}/train_visualization.png"),
        "Test Sequence Visualization": wandb.Image(f"{save_dir}/test_visualization.png"),
        })
    
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_samples', type=int, default=30000)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_outputs', type=int, default=1)
    args = parser.parse_args()

    main(args)