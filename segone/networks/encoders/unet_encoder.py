import torch.nn as nn

from segone.utils.common_layers import Bottleneck, ConvBlock


class UNetEncoder(nn.Module):
    def __init__(self, opts):
        """BottleNeck + MaxPool > Conv > Conv"""
        super(UNetEncoder, self).__init__()

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
            self.convs[f"conv_1_{i}"] = ConvBlock(self.channels[i], self.channels[i + 1])
            self.convs[f"conv_2_{i}"] = ConvBlock(self.channels[i + 1], self.channels[i + 1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def get_channels(self):
        """Returns encoded channels for skip connections"""
        return self.channels

    def forward(self, x):
        self.features = []

        x = self.bottleneck(x)
        self.features.append(x)
        for i in range(self.opts["num_layers"]):
            x = self.pool(x)
            x = self.convs[f"conv_1_{i}"](x)
            x = self.convs[f"conv_2_{i}"](x)

            self.features.append(x)

        return self.features
