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


class ClaDecoder(nn.Module):
    def __init__(self, opts, channel_enc):
        super(ClaDecoder, self).__init__()

        self.opts = opts
        self.channel_enc = channel_enc
        self.len_ch_enc = len(self.channel_enc)

        self.conv = Conv1DBlock(
            kernel_size=self.channel_enc[-1],
            out_channels=self.channel_enc[-1],
            stride = self.channel_enc[-1],
            use_relu=False,
            use_pad=False
        )

        self.head = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channel_enc[-1], self.opts["channel_out"])

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]
        x, (b, c, h, w) = spatial_flatten(x)
        x = self.conv(x)
        x = channel_recover(x, h, w)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
