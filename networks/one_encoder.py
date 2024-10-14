import sys
sys.path.insert(0,'/home/sang/Desktop/Conv1D/segone')

import torch
import torch.nn as nn

from utils.layers import *

class BottleneckBlock(nn.Module):
    def __init__(self, scale=2):
        super(BottleneckBlock, self).__init__()

        self.scale = scale
        self.kernel_size = self.scale * 2 - 1

        self.unshuffle = PixelUnshuffle(scale=self.scale)
        self.shuffle = PixelShuffle(scale=self.scale)
        self.conv_down = Conv1DBlock(kernel_size=self.kernel_size)
        self.conv_up = Conv1DBlock(kernel_size=3)
        self.upsample = UpScale1D(self.scale)

    def forward(self, x):
        x = self.unshuffle(x)
        x, (b, c, h, w) = spatial_flatten(x)
        x = self.upsample(x)
        x = self.conv_down(x)
        x = spatial_recover(x, (b, c*self.scale, h, w))
        x = self.shuffle(x)
        x, (b, c, h, w) = spatial_flatten(x)
        x = self.conv_up(x)
        x = spatial_recover(x, (b, c, h, w))

        return x
    

class Bottleneck(nn.Module):
    def __init__(self, scale=2, repeat=3):
        super(Bottleneck, self).__init__()

        self.scale = scale
        self.kernel_size = self.scale * 2 - 1

        self.blocks = nn.ModuleList([
            BottleneckBlock(self.scale)
            for _ in range(repeat)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class OneEncoder(nn.Module):
    def __init__(self, channel_in, num_layers=5, num_conv=1, kernel_size=3, pool_kernel_size=3):
        super(OneEncoder, self).__init__()

        self.channel_in = channel_in
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Calculate channel outs
        self.channels = [(2 ** i) * channel_in for i in range(num_layers+1)]
        self.channels.reverse()

        # Initialize Layers
        self.convs = nn.ModuleList([
            Conv1DBlock(kernel_size=self.kernel_size)
            for _ in range(self.num_layers)
        ])
        self.downsample = PixelUnshuffle(scale=2)
        self.pool = Pool1D(kernel_size=self.pool_kernel_size)

    def get_channels(self):
        """Returns encoded channels (deep to shallow) for skip connections
        """
        return self.channels

    def forward(self, x):
        self.features = []
        self.features.append(x)

        for i in range(self.num_layers):
            self.features.append(self.downsample(self.features[-1]))
            self.features[-1] = self.downsample(self.features[-1])
            self.features[-1] = spatial_flatten(self.features[-1])
            self.features[-1] = self.pool(self.features[-1])
            self.features[-1] = self.convs[i](self.features[-1])
            self.features[-1] = spatial_recover(self.features[-1])
        
        return self.features
    
sample = torch.ones((5,3,512,512))
model = Bottleneck(repeat=1)
out = model(sample)
arr = out[0].detach().numpy()
print(arr.shape)