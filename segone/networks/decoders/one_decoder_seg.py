import torch
import torch.nn as nn

from segone.utils.layers import (
    Conv1DBlock,
    PixelShuffle,
    PixelUnshuffle,
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
        self.bottleneck = Conv1DBlock(
            kernel_size=self.channel_enc[-1],
            out_channels=self.channel_enc[-1] // 2,
            stride=self.channel_enc[-1],
            use_pad=False,
        )

        self.convs = nn.ModuleDict()
        for i in range(self.len_ch_enc - 1, 0, -1):
            # self.convs[f"spatial_{i}"] = Conv1DBlock(kernel_size=self.opts["kernel_size"])
            self.convs[f"channel_1_{i}"] = Conv1DBlock(
                kernel_size=int(2.5 * self.channel_enc[i]) if i > 1 else self.channel_enc[i] // 2,
                out_channels=self.channel_enc[i] if i > 1 else self.channel_enc[i] // 2,
                stride=int(2.5 * self.channel_enc[i]) if i > 1 else self.channel_enc[i] // 2,
                use_relu=i > 1,
                use_bn=i > 1,
                use_pad=False,
            )
            self.convs[f"channel_2_{i}"] = Conv1DBlock(
                kernel_size=1,
                in_channels=self.channel_enc[i] if i > 1 else self.channel_enc[i] // 2,
                out_channels=self.channel_enc[i] if i > 1 else self.channel_enc[i] // 2,
                stride=1,
                use_relu=i > 1,
                use_bn=i > 1,
                use_pad=False,
            )
            if i == 1:
                self.convs[f"head_{i}"] = Conv1DBlock(
                    kernel_size=1,
                    in_channels=self.channel_enc[i - 1] // 2 if i > 1 else self.channel_enc[i - 1] // 4,
                    out_channels=self.opts["channel_out"],
                    stride=1,
                    use_relu=False,
                    use_bn=False,
                    use_pad=False,
                )

        self.downsample = PixelUnshuffle(scale=2)
        self.upsample = PixelShuffle(scale=2)

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]
        x, (_, _, h, w) = spatial_flatten(x)
        x = self.bottleneck(x)
        x = channel_recover(x, h, w)
        for i in range(self.len_ch_enc - 1, 0, -1):
            x = [x]
            if i > 1:  # play with 0/1
                x += [self.downsample(features_enc[i - 1])]
            x = torch.cat(x, 1)
            x, (_, _, h, w) = spatial_flatten(x)
            # x = self.convs[f"spatial_{i}"](x)
            x = self.convs[f"channel_1_{i}"](x)
            x = self.convs[f"channel_2_{i}"](x)
            x = channel_recover(x, h, w)
            x = self.upsample(x)
            if i == 1:
                _, _, h, w = x.size()
                pred = torch.flatten(x, start_dim=2)
                self.outputs.append(channel_recover(self.convs[f"head_{i}"](pred), h, w))

        return self.outputs
