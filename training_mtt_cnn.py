# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:45:38 2025

@author: dpqb1
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from mtt_cnn import MTTModel
from synth_xray_data import MTTSyntheticDataset, generate_fn

# Assume MTTModel is imported or defined elsewhere with output_channels=3
num_outputs = 1
model = MTTModel(input_channels=1, output_channels=num_outputs).to('cuda' if torch.cuda.is_available() else 'cpu')

# Create Dataset instance
num_train_samples = 1
num_test_samples = 1
training_dataset = MTTSyntheticDataset(num_samples=num_train_samples,
                              sequence_length=1,
                              input_shape=(1, 32, 96), 
                              generate_fn=generate_fn,
                              start_idx=0)

test_dataset = MTTSyntheticDataset(num_samples=num_test_samples,
                              sequence_length=30,
                              input_shape=(1, 32, 96), 
                              generate_fn=generate_fn,
                              start_idx=num_train_samples)

# Dataset and DataLoader
batch_size = 1
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Loss function: Mean Squared Error
criterion = nn.MSELoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training loop params
num_epochs = 60
train_losses = []
val_losses = []

#%% Training Loop 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, truths) in enumerate(train_loader):
        inputs = inputs.to(device)      # shape (B, 1, H, W)
        truths = truths.to(device)      # shape (B, num_outputs, H, W)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)         # shape (B, num_outputs, H, W)
        loss = criterion(outputs, truths)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] average loss: {epoch_loss:.6f}")
    
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for inputs, truths in val_loader:
    #         inputs, truths = inputs.to(device), truths.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, truths)
    #         val_loss += loss.item()

    # val_loss /= len(val_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}] VAL loss: {val_loss:.6f}")
    
    # # After each epoch:
    train_losses.append(epoch_loss)
    # val_losses.append(val_loss)
    

for batch_idx, (inputs, truths) in enumerate(train_loader):
    inputs = inputs.to(device)      # shape (B, 1, H, W)
    truths = truths.to(device)      # shape (B, 4, H, W)
    model.eval()
    outputs = model(inputs)         # shape (B, 4, H, W)
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

print("Training complete!")
plt.figure()
plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

#%% Test
'''
import numpy as np
from matplotlib import pyplot as plt
threshold = 0

# Create your test dataset
test_dataset = MTTSyntheticDataset(num_samples=1,
                                   sequence_length=30,
                                   input_shape=(1, 32, 1024), 
                                   generate_fn=generate_fn,
                                   start_idx=num_train_samples)

# Create a DataLoader for batching (optional, can do batch_size=1 for simple test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set model to evaluation mode
model.eval()

# Disable gradient calculations for inference
with torch.no_grad():
    for inputs, truths in test_loader:
        inputs = inputs.to(device)
        truths = truths.to(device)

        outputs = model(inputs)
        
        # Optionally compute test loss or metrics here
        loss = criterion(outputs, truths)
        print(f"Test batch loss: {loss.item():.4f}")
        
        frame = inputs[0, 0, :, :].cpu().numpy()
        gt = truths[0, 0, :, :].cpu().numpy()
        pred = outputs[0, 0, :, :].detach().cpu().numpy()
        
        # Get coordinates
        gt_yx = np.argwhere(gt > 0.5)
        pred_yx = np.argwhere(pred > threshold)
        
        # Plot
        plt.imshow(frame, cmap='gray')
        plt.scatter(gt_yx[:,1], gt_yx[:,0], color='lime', label='GT', s=40, marker='o')
        plt.scatter(pred_yx[:,1], pred_yx[:,0], color='red', label='Pred', s=20, marker='x')
        # plt.title(f'Frame {t}')
        plt.legend()
        plt.show()
        
        # Optionally compute test loss or metrics here
        loss = criterion(outputs, truths)
        print(f"Test batch loss: {loss.item():.4f}")

        # Optional: visualize or further process outputs vs truths
        break  # Remove break if you want to run all batches

'''