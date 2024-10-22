import torch
import torch.nn as nn

from segone.networks.encoders.resnet_encoder import *
from segone.networks.encoders.skipinit_encoder import *
from segone.networks.encoders.eunnet_encoder import *
from segone.networks.encoders.unet_encoder import *
from segone.networks.decoders.common_decoder_seg import *
from segone.networks.decoders.common_decoder_cla import *
from segone.utils.layers import *


class CommonNet(nn.Module):
    def __init__(self, opts):
        super(CommonNet, self).__init__()
        self.opts = opts

        self.encoder = OneEncoder(self.opts)
        if self.opts["type"] == "segmentation":
            self.decoder = SegDecoder(self.opts, channel_enc=self.encoder.get_channels())

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
