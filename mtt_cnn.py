# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:20:54 2025

@author: dpqb1
"""

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert

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
    def __init__(self, input_channels=1, output_channels=1):  # output_channels=4 now
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
    
class MTTModelQAT(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):  # output_channels=4 now
        super(MTTModelQAT, self).__init__()    
        self.quant = QuantStub()
        
        # Encoder remains unchanged
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        
        # Bottleneck remains unchanged
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters1, num_filters2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters2, num_filters2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters2, num_filters2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters2, num_filters1, kernel_size=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters1, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        )
        
        self.dequant = DeQuantStub()
        
        
    def forward(self, x):
        x = self.quant(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        # Fuse conv + relu layers in encoder
        for idx in range(0, len(self.encoder), 2):
            fuse_modules(self.encoder, [str(idx), str(idx + 1)], inplace=True)
    
        # Fuse conv + relu layers in bottleneck
        for idx in range(0, len(self.bottleneck), 2):
            fuse_modules(self.bottleneck, [str(idx), str(idx + 1)], inplace=True)
    
        # Fuse convtranspose + relu layers in decoder except the last convtranspose
        # for idx in range(0, len(self.decoder) - 1, 2):
        #     fuse_modules(self.decoder, [str(idx), str(idx + 1)], inplace=True)


if __name__ == "__main__":
    model = MTTModel(input_channels=1, output_channels=4).cuda()
    print(model)
    
    # Create dummy input: batch_size=1, channels=1, 128x128 image
    dummy_input = torch.randn(1, 1, 128, 128).cuda()
    
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)