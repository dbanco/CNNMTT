# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:20:54 2025

@author: dpqb1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def mtt_loss_mse(pred, truth):
    """
    MSE loss applied to all 4 channels equally.
    Matches the paper's expected average squared error objective.
    """
    mse_loss = nn.MSELoss()
    return mse_loss(pred, truth)

kernel_size = 9
num_filters1 = 64
num_filters2 = 512
stride = 2
padding = 4
output_padding = 1
class MTTModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):  # output_channels=4 now
        super(MTTModel, self).__init__()
        
        # Encoder remains unchanged
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.01),
            nn.Conv2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.01),
            nn.Conv2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.01)
        )
        
        # Bottleneck remains unchanged
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters1, num_filters2, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(num_filters2, num_filters2, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(num_filters2, num_filters2, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(num_filters2, num_filters1, kernel_size=1),
            nn.LeakyReLU(0.01)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(num_filters1, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
)
        
    def forward(self, x):
        x_enc = self.encoder(x)
        x_bottleneck = self.bottleneck(x_enc)
        out = self.decoder(x_bottleneck)
        
        return out


if __name__ == "__main__":
    model = MTTModel(input_channels=1, output_channels=4).cuda()
    print(model)
    
    # Create dummy input: batch_size=1, channels=1, 128x128 image
    dummy_input = torch.randn(1, 1, 128, 128).cuda()
    
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)