import argparse
import time

import torch
from torchinfo import summary

from segone.networks.common_network import CommonNet
from segone.networks.segone_network import SegOne

tests = [
(
    "OneNet_e,4 Segmentation",
    SegOne,
    {
        "name": "ONENET",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 4,
        "bottleneck_scale": 1,
        "bottleneck_repeat": 3,
        "bottleneck_channel": 64,
        "kernel_size": 9,
    }
),
(
    "OneNet_ed,4 Segmentation",
    SegOne,
    {
        "name": "SEGONE",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 4,
        "bottleneck_scale": 1,
        "bottleneck_repeat": 3,
        "bottleneck_channel": 64,
        "kernel_size": 9,
    }
),
(
    "Resnet34 Segmentation",
    CommonNet,
    {
        "name": "RESNET",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 34,
    }
),
(
    "Resnet50 Segmentation",
    CommonNet,
    {
        "name": "RESNET",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 50,
    }
),
(
    "UNet_4 Segmentation",
    CommonNet,
    {
        "name": "UNET",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 4,
        "bottleneck_repeat": 2,
        "bottleneck_channel": 64,
    }
),
(
    "MobileNet Segmentation",
    CommonNet,
    {
        "name": "MOBILENET",
        "type": "segmentation",
        "channel_in": 3,
        "channel_out": 4,
        "num_layers": 4,
        "bottleneck_repeat": 3,
        "bottleneck_channel": 64,
    }
),
]


if __name__ == "__main__":
    """Test dummy inputs for structural testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.cuda}" if args.cuda > -1 else "cpu"

    input_size=(1,3,256,256)

    for model_name, model_type, opts in tests:
        print(model_name)
        model = model_type(opts).to(device=device)

        torch.cuda.synchronize()
        start_time = time.time()

        summary(model, input_size=input_size, device=device, verbose=args.verbose)

        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Runtime: {end_time-start_time:.4f}s")

        print("\n\n\n" + "".join(["#"] * 95) + "\n\n\n")
