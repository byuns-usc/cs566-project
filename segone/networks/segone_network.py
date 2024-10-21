import torch
import torch.nn as nn

from segone.utils.layers import *


class SegOne(nn.Module):
    def __init__(self, opts):
        super(SegOne, self).__init__()
        self.opts = opts

    def forward(self, x):
        return x
