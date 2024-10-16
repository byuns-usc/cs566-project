import torch
import torch.nn as nn

from segone.utils.layers import (
    Conv1DBlock,
    PixelShuffle,
    channel_flatten,
    channel_recover,
    spatial_flatten,
    spatial_recover,
)


class SegDecoder(nn.Module):
    def __init__(self, opts, channel_enc):
        super(SegDecoder, self).__init__()

        self.opts = opts
        self.channel_enc = channel_enc
        self.len_ch_enc = len(self.channel_enc)

        # Initialize Layers
        self.convs = nn.ModuleDict()
        for i in range(self.len_ch_enc - 1, 0, -1):
            self.convs[f"channel_1_{i}"] = Conv1DBlock(
                kernel_size=self.channel_enc[i], out_channels=2 * self.channel_enc[i], stride=self.channel_enc[i]
            )
            self.convs[f"channel_2_{i}"] = Conv1DBlock(
                kernel_size=2 * self.channel_enc[i - 1],
                out_channels=2 * self.channel_enc[i - 1],
                stride=2 * self.channel_enc[i - 1],
            )
            self.convs[f"spatial_{i}"] = Conv1DBlock(kernel_size=self.opts["kernel_size"])
            self.convs[f"head_{i}"] = Conv1DBlock(
                kernel_size=2 * self.channel_enc[i - 1],
                out_channels=self.opts["channel_out"],
                stride=2 * self.channel_enc[i - 1],
            )

        self.upsample = PixelShuffle(scale=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]
        x, _ = spatial_flatten(x)
        for i in range(self.len_ch_enc - 1, -1, -1):
            x = self.convs[f"channel_1_{i}"](x)
            x = channel_recover(x, h, w)
            x = self.upsample(x)
            x, (_, _, h, w) = spatial_flatten(x)
            x = self.convs[f"channel_2_{i}"](x)
            x = channel_flatten(x)
            self.outputs.insert(0, self.channel_recover(self.convs[f"head_{i}"](x), h, w))
            x = self.convs[f"spatial_{i}"](x)

        return self.outputs
