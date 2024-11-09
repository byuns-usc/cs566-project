import time

import torch
import torch.nn as nn

from segone.networks.decoders.common_decoder_seg import CommonSegDecoder

from segone.networks.decoders.segone_decoder_seg import SegDecoder
# from segone.networks.decoders.one_decoder_seg import SegDecoder
from segone.networks.decoders.segone_decoder_cla import ClaDecoder
from segone.networks.encoders.one_encoder import OneEncoder


class SegOne(nn.Module):

    model_types = {
        "SEGONE": {"segmentation": SegDecoder, "classification": ClaDecoder},
        "ONENET": {"segmentation": CommonSegDecoder, "classification": ClaDecoder},
    }

    def __init__(self, opts):
        super(SegOne, self).__init__()
        self.opts = opts

        assert self.opts["name"] in self.model_types
        assert self.opts["type"] in self.model_types[self.opts["name"]]

        self.encoder = OneEncoder(self.opts)
        self.decoder = self.model_types[self.opts["name"]][self.opts["type"]](
            self.opts, channel_enc=self.encoder.get_channels()
        )

    def forward(self, x):
        # torch.cuda.synchronize()
        # start_time = time.time()
        # enc_features = self.encoder(x)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Encoder runtime: {end_time-start_time:.4f}s")
        # torch.cuda.synchronize()
        # start_time = time.time()
        # outputs = self.decoder(enc_features)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Decoder runtime: {end_time-start_time:.4f}s")
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
