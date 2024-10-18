import torch
import torch.nn as nn

from segone.utils.layers import (
    Conv1DBlock,
    PixelShuffle,
    PixelUnshuffle,
    Pool1D,
    UpScale1D,
    channel_flatten,
    channel_recover,
    spatial_flatten,
    spatial_recover,
)


class BottleneckBlock(nn.Module):
    """Performs single spatial conv1D"""

    def __init__(self, scale=2):
        super(BottleneckBlock, self).__init__()

        self.scale = scale
        self.kernel_size = self.scale * 2 + 1

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
        x = spatial_recover(x, (b, c * self.scale, h, w))
        x = self.shuffle(x)
        x, (b, c, h, w) = spatial_flatten(x)
        x = self.conv_up(x)
        x = spatial_recover(x, (b, c, h, w))

        return x


class Bottleneck(nn.Module):
    """Chain of Conv1D for channel expansion"""

    def __init__(self, scale=2, repeat=3, in_channels=3, out_channels=32):
        super(Bottleneck, self).__init__()

        self.scale = scale
        self.kernel_size = self.scale * 2 - 1

        # self.blocks = nn.ModuleList([
        #     BottleneckBlock(self.scale)
        #     for _ in range(repeat)
        # ])

        self.blocks = nn.ModuleList(
            [
                Conv1DBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=in_channels if i == 0 else out_channels,
                )
                for i in range(repeat)
            ]
        )

    def forward(self, x):
        # for block in self.blocks:
        #     x = block(x)

        x, (b, c, h, w) = spatial_flatten(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i < len(self.blocks) - 1:
                x = channel_flatten(x)
        x = channel_recover(x, h, w)
        return x


class OneEncoder(nn.Module):
    def __init__(self, opts):
        """PUD > Conv1D(k) > Conv1D(C,C/2) > Conv1D(C/2,C/2)"""
        super(OneEncoder, self).__init__()

        self.opts = opts

        # Calculate channel outs
        self.channels = [(2**i) * self.opts["channel_in"] for i in range(self.opts["num_layers"] + 1)]

        # Initialize Layers
        self.bottleneck = Bottleneck(scale=self.opts["bottleneck_scale"], repeat=self.opts["bottleneck_repeat"])
        self.convs = nn.ModuleDict()
        for i in range(self.opts["num_layers"]):
            self.convs[f"spatial_{i}"] = Conv1DBlock(kernel_size=self.opts["kernel_size"])
            self.convs[f"channel_1_{i}"] = Conv1DBlock(
                kernel_size=2 * self.channels[i + 1], out_channels=self.channels[i + 1], stride=2 * self.channels[i + 1]
            )
            self.convs[f"channel_2_{i}"] = Conv1DBlock(
                kernel_size=self.channels[i + 1], out_channels=self.channels[i + 1], stride=self.channels[i + 1]
            )
        self.downsample = PixelUnshuffle(scale=2)
        # self.pool = Pool1D(kernel_size=self.opts['pool_kernel_size'])

    def get_channels(self):
        """Returns encoded channels for skip connections"""
        return self.channels

    def forward(self, x):
        self.features = []

        x = self.bottleneck(x)
        self.features.append(x)

        for i in range(self.opts["num_layers"]):
            x = self.downsample(x)
            x, (_, _, h, w) = spatial_flatten(x)
            x = self.convs[f"spatial_{i}"](x)
            x = self.convs[f"channel_1_{i}"](x)
            x = channel_flatten(x)
            x = self.convs[f"channel_2_{i}"](x)
            x = channel_recover(x, h, w)

            self.features.append(x)

        return self.features
