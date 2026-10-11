"""
Project: Road-D-Loc (Road Defect Localization and Quantification)
Module 2: Deep Segmentation Architecture (U-Net)
Author: Tarun Yadav
Registration No: 24BAI10576
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function for the main convolutional block
def conv_block(in_channels, out_channels):
    """A standard two-layer convolutional block used in U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    This model performs pixel-level defect localization.
    
    Classes:
    - 0: Background
    - 1: Pothole
    - 2: Crack
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        # Output layer (Background, Pothole, and Crack classes)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Downsampling
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, kernel_size=2, stride=2)
        
        e4 = self.enc4(p3)
        p4 = F.max_pool2d(e4, kernel_size=2, stride=2)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Upsampling and Concatenation (Copy & Crop)
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        # Output
        return self.out_conv(d1)

def load_model(path="./models/checkpoints/best_unet_weights.pth", num_classes=3, device="cpu"):
    """
    Initializes U-Net model and loads weights if checkpoint exists.
    """
    model = UNet(in_channels=3, out_channels=num_classes)
    
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"INFO: Successfully loaded model weights from {path}")
        except Exception as e:
            print(f"WARNING: Could not load weights from {path}: {e}. Using initialized model.")
    else:
        print(f"INFO: Model initialized with {num_classes} output classes. (Weights not found at {path}, mock/demo inference available)")
        
    model.to(device)
    model.eval()
    return model
