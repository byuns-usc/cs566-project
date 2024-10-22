import torch
import torch.nn as nn

from segone.networks.encoders.segone_encoder import OneEncoder
from segone.networks.decoders.segone_decoder_seg import SegDecoder
from segone.networks.decoders.segone_decoder_cla import *
from segone.utils.layers import *


class SegOne(nn.Module):
    def __init__(self, opts):
        super(SegOne, self).__init__()
        self.opts = opts

        self.encoder = OneEncoder(self.opts)
        if self.opts["type"] == "segmentation":
            self.decoder = SegDecoder(self.opts, channel_enc=self.encoder.get_channels())

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
