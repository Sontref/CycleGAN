import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# c7s1-k denotes a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1. 
# dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.

# Reflection padding was used to reduce artifacts.

# Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layers. 
# uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2.

# The network with 6 residual blocks consists of:
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual=6):
        super().__init__()

        # c7s1-64
        conv1 = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # d128
        down1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # d256
        down2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        
        res_layer = []
        for _ in range(num_residual):
            # R256
            res_layer += [ResBlock()]
        res_layer = nn.Sequential(*res_layer)

        # u128
        up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # u64
        up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        #c7s1-3
        conv2 = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.generator = nn.Sequential(conv1, down1, down2, res_layer, up1, up2, conv2)

    def forward(self, x): 
        return self.generator(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels=256, kernel_size=3):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels,
                kernel_size=kernel_size
                ),
            nn.InstanceNorm2d(in_channels),
        )
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels,
                kernel_size=kernel_size
            ),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        c1 = self.layer1(x)
        c1 = F.relu(c1)
        c2 = self.layer2(c1)

        return x + c2

# For discriminator networks, we use 70 × 70 PatchGAN. 
# Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 
# After the last layer, we apply a convolution to produce a 1-dimensional output. 
# We do not use InstanceNorm for the first C64 layer. 
# We use leaky ReLUs with a slope of 0.2. 
# The discriminator architecture is: C64-C128-C256-C512

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.discriminator = nn.Sequential(conv1, conv2, conv3, conv4)

    def forward(self, x):
        return self.discriminator(x)


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            Generator(),
            Discriminator()
        )
    
    def forward(self, x):
        return self.model(x)