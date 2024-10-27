import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, channel_in, channel_out, repeat=3):
        super(Bottleneck, self).__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(channel_in if i == 0 else channel_out, channel_out, kernel_size=3, padding=1)
                for i in range(repeat)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")
