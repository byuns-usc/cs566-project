import torch.nn as nn

from segone.networks.decoders.common_decoder_seg import CommonSegDecoder
from segone.networks.encoders.mobilenet_encoder import MobileNetEncoder
from segone.networks.encoders.resnet_encoder import ResNetEncoder
from segone.networks.encoders.unet_encoder import UNetEncoder


class CommonNet(nn.Module):

    encoder_types = {
        "RESNET": ResNetEncoder,
        "UNET": UNetEncoder,
        "MOBILENET": MobileNetEncoder,
        # "SKIPINIT": SkipinitEncoder,
        # "EUNNET": EUNEncoder,
    }

    decoder_types = {"segmentation": CommonSegDecoder, "classification": CommonSegDecoder}

    def __init__(self, opts):
        super(CommonNet, self).__init__()
        self.opts = opts

        assert self.opts["name"] in self.encoder_types
        assert self.opts["type"] in self.decoder_types

        self.encoder = self.encoder_types[self.opts["name"]](self.opts)
        self.decoder = self.decoder_types[self.opts["type"]](self.opts, channel_enc=self.encoder.get_channels())

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
