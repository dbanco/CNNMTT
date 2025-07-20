# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:52:09 2025

@author: dpqb1
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

from mtt_cnn import MTTModel
from synth_xray_data import MTTSyntheticDataset, generate_fn

import logging
import os
from datetime import datetime

import wandb

def train(model, train_loader, criterion, optimizer, device, num_epochs):
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
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, truths) in enumerate(train_loader):
            inputs, truths = inputs.to(device), truths.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, truths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] average loss: {epoch_loss:.6f}")
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss})
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss
        }, checkpoint_path)
        
        logging.info(f"Saved checkpoint: {checkpoint_path}")

    return train_losses

def visualize_results(model, loader, device, num_outputs):
    model.eval()
    for batch_idx, (inputs, truths) in enumerate(loader):
        inputs, truths = inputs.to(device), truths.to(device)
        outputs = model(inputs)

        fig, axs = plt.subplots(3, num_outputs, figsize=(12, 9))
        for i in range(num_outputs):
            axs[0].imshow(inputs.cpu().numpy()[0, 0], cmap='gray')
            axs[1].imshow(truths.cpu().numpy()[0, i], cmap='viridis')
            axs[2].imshow(outputs.cpu().detach().numpy()[0, i], cmap='viridis')
            axs[0].set_title(f'Ch {i}')
        axs[0].set_ylabel("Input")
        axs[1].set_ylabel("Truth")
        axs[2].set_ylabel("Output")
        plt.tight_layout()
        plt.show()
        break

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTTModel(input_channels=1, output_channels=args.num_outputs).to(device)

    train_dataset = MTTSyntheticDataset(num_samples=args.num_train_samples,
                                        sequence_length=args.sequence_length,
                                        input_shape=(1, args.height, args.width),
                                        generate_fn=generate_fn,
                                        start_idx=0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, pin_memory=True, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    wandb.init(project="mtt-cnn", config={
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "architecture": "MTTModel"
    })
    wandb.watch(model, log="all")

    train_losses = train(model, train_loader, criterion, optimizer, device, args.num_epochs)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    visualize_results(model, train_loader, device, args.num_outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_samples', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=1)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_outputs', type=int, default=1)
    args = parser.parse_args()

    main(args)