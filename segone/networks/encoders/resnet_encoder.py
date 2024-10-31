import numpy as np
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50

from segone.utils.common_layers import Bottleneck, ConvBlock


class ResNetEncoder(nn.Module):
    """Resnet 18/34/50 backbone encoder"""

    def __init__(self, opts):
        """BottleNeck + MaxPool > Conv > Conv"""
        super(ResNetEncoder, self).__init__()

        self.opts = opts

        # Calculate channel outs
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # Initialize Layers
        resnets = {18: resnet18, 34: resnet34, 50: resnet50}
        weights = {
            18: ResNet18_Weights.IMAGENET1K_V1,
            34: ResNet34_Weights.IMAGENET1K_V1,
            50: ResNet50_Weights.IMAGENET1K_V2,
        }

        assert self.opts["num_layers"] in resnets

        self.encoder = resnets[self.opts["num_layers"]](weights=weights[self.opts["num_layers"]])

        if self.opts["num_layers"] > 34:
            self.num_ch_enc[1:] *= 4

    def get_channels(self):
        """Returns encoded channels for skip connections"""
        return self.num_ch_enc

    def forward(self, x):
        self.features = []
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
