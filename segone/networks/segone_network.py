import torch
import torch.nn as nn

from segone.networks.encoders.segone_encoder import OneEncoder
from segone.networks.decoders.segone_decoder_seg import SegDecoder
from segone.networks.decoders.segone_decoder_cla import ClaDecoder
from segone.utils.layers import *


class SegOne(nn.Module):

    decoder_types = {"segmentation": SegDecoder, "classification": ClaDecoder}

    def __init__(self, opts):
        super(SegOne, self).__init__()
        self.opts = opts

        assert self.opts["type"] in self.decoder_types

        self.encoder = OneEncoder(self.opts)
        self.decoder = self.decoder_types[self.opts["type"]](self.opts, channel_enc=self.encoder.get_channels())

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
