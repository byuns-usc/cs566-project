import torch
import torch.nn as nn

from segone.utils.common_layers import ConvBlock, upsample


class CommonSegDecoder(nn.Module):
    def __init__(self, opts, channel_enc):
        super(CommonSegDecoder, self).__init__()

        self.opts = opts
        self.channel_enc = channel_enc
        self.len_ch_enc = len(self.channel_enc)

        # Initialize Layers
        self.convs = nn.ModuleDict()
        for i in range(self.len_ch_enc - 1, 0, -1):
            self.convs[f"channel_1_{i}"] = ConvBlock(self.channel_enc[i], self.channel_enc[i - 1])

            self.convs[f"channel_2_{i}"] = ConvBlock(self.channel_enc[i - 1] * 2, self.channel_enc[i - 1])

            self.convs[f"head_{i}"] = ConvBlock(self.channel_enc[i - 1], self.opts["channel_out"])

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]

        for i in range(self.len_ch_enc - 1, 0, -1):
            x = self.convs[f"channel_1_{i}"](x)
            x = [upsample(x)]
            x += [features_enc[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[f"channel_2_{i}"](x)
            self.outputs.append(self.convs[f"head_{i}"](x))

        return self.outputs
