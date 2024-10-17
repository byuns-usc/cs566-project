import torch
import torch.nn as nn

from segone.utils.layers import *


class Bottleneck2D(nn.Module):
    def __init__(self, channel=3, repeat=3):
        super(Bottleneck2D, self).__init__()
        self.blocks = nn.ModuleList(
            [nn.Conv2d(channel * (2**i), channel * (2 ** (i + 1)), kernel_size=3, padding=1) for i in range(repeat)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
