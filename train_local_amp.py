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

from torch.cuda.amp import GradScaler, autocast

import argparse
from mtt_cnn import MTTModel
from synth_xray_data import MTTSyntheticDataset, PreloadedMTTDataset

import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, num_epochs, use_amp):
    train_losses = []
    scaler = GradScaler() if use_amp else None

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
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, truths) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, truths)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, truths)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        # Save checkpoint
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] average loss: {epoch_loss:.6f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss
        }, checkpoint_path)

        logging.info(f"Saved checkpoint: {checkpoint_path}")

    return train_losses

def visualize_sequence(model, data_loader, save_dir, prefix="sequence", time_indices=[0,4,8,12,16,24,28]):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        inputs, truths = next(iter(data_loader))  # (T, 1, H, W)

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


def main(args):
    model = MTTModel(input_channels=1, output_channels=args.num_outputs)
    
    train_dataset = PreloadedMTTDataset("mtt_dataset_10x30x32x96.pt")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    train(model, train_loader, criterion, optimizer, args.num_epochs,args.use_amp)


    test_dataset = MTTSyntheticDataset(num_spots=3,
                                        num_samples=args.num_train_samples,
                                        sequence_length=args.sequence_length,
                                        noise=True,
                                        input_shape=(1, args.height, args.width),
                                        seed=2,
                                        preload=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.sequence_length, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.sequence_length, shuffle=False)


    save_dir = "./visualizations"
    visualize_sequence(model, train_loader, save_dir, prefix="train")
    visualize_sequence(model, test_loader, save_dir, prefix="test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_samples', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_outputs', type=int, default=1)
    parser.add_argument('--use_amp', type=bool, default=True)
    args = parser.parse_args()
        
    main(args)