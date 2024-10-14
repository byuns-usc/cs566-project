import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# B x C x H x W to B x H x W x C to B x HWC
def spatial_flatten(x):
    """Flattens to Bx1xHWC and returns tensor with original b,c,h,w values
    """
    ratio = x.size()
    x = torch.permute(x, (0, 2, 3, 1))
    x = x.reshape(x.size()[0], 1, -1)
    return x, ratio

# B x HWC to B x H x W x C to B x C x H x W
def spatial_recover(x, ratio):
    """Recovers flattened shape of Bx1xHWC to given b,c,h,w ratio
    """
    x = x.reshape(ratio[0], ratio[2], ratio[3], ratio[1])
    x = torch.permute(x, (0, 3, 1, 2))
    return x

# Conv1D
class Conv1DBlock(nn.Module):
    """Layer to pad and convolve input Bx1xHWC -> Bx1xHWC
    """
    def __init__(self, kernel_size, use_refl=True):
        super(Conv1DBlock, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad1d(kernel_size//2)
        else:
            self.pad = nn.ZeroPad1d(kernel_size//2)
        self.conv = nn.Conv1d(1, 1, kernel_size, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(x.size())
        x = self.pad(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

# MaxPool
class Pool1D(nn.Module):
    """Layer to maxpool Bx1xHWC -> Bx1xHWC/2
    """
    def __init__(self, kernel_size):
        super(Pool1D, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=2, padding=kernel_size//2)

    def forward(self, x):
        x = self.pool(x)
        return x

# Upscaling, call Conv1D afterwards
class UpScale1D(nn.Module):
    """Upsamples channel dim by Bx1xHWC -> Bx1x2HWC
    """
    def __init__(self, scale):
        super(UpScale1D, self).__init__()
        self.upscale = nn.Upsample(scale_factor=scale, mode='linear')

    def forward(self, x):
        x = self.upscale(x)
        return x

# PixelShuffle
class PixelShuffle(nn.Module):
    """Upsamples image by BxCxHxW -> BxC/4x2Hx2W
    """
    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.shuffle(x)
        return x

# PixelUnshuffle
class PixelUnshuffle(nn.Module):
    """Upsamples image by BxCxHxW -> BxC/4x2Hx2W
    """
    def __init__(self, scale):
        super(PixelUnshuffle, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x):
        x = self.unshuffle(x)
        return x

# Losses
