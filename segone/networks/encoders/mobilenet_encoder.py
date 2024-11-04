import numpy as np
import torch.nn as nn

from segone.utils.common_layers import Bottleneck


class MobileBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super(MobileBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, groups=in_channels, padding="same")
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class MobileNetEncoder(nn.Module):
    """Resnet 18/34/50 backbone encoder"""

    def __init__(self, opts):
        super(MobileNetEncoder, self).__init__()

        self.opts = opts

        # Calculate channel outs
        self.channels = [(2**i) * self.opts["bottleneck_channel"] for i in range(self.opts["num_layers"] + 1)]

        # Initialize Layers
        self.bottleneck = Bottleneck(
            self.opts["channel_in"],
            self.opts["bottleneck_channel"],
            repeat=self.opts["bottleneck_repeat"],
        )
        self.convs = nn.ModuleDict()
        for i in range(self.opts["num_layers"]):
            self.convs[f"conv_1_{i}"] = MobileBlock(self.channels[i], self.channels[i + 1])
            self.convs[f"conv_2_{i}"] = MobileBlock(self.channels[i + 1], self.channels[i + 1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def get_channels(self):
        """Returns encoded channels for skip connections"""
        return self.channels

    def forward(self, x):
        self.features = []

        x = (x - 0.45) / 0.225
        x = self.bottleneck(x)
        self.features.append(x)
        for i in range(self.opts["num_layers"]):
            x = self.pool(x)
            x = self.convs[f"conv_1_{i}"](x)
            x = self.convs[f"conv_2_{i}"](x)

            self.features.append(x)

        return self.features
