import argparse
import os
import time

import torch
from torchinfo import summary

from segone.networks.common_network import CommonNet
from segone.networks.segone_network import SegOne

if __name__ == "__main__":
    """Test dummy inputs for structural testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.cuda}" if args.cuda > -1 else "cpu"

    print("Verify SegOne Classification Architecture")
    model = SegOne(
        {
            "type": "classification",
            "channel_in": 3,
            "channel_out": 13,
            "num_layers": 4,
            "bottleneck_scale": 2,
            "bottleneck_repeat": 3,
            "bottleneck_channel": 32,
            "kernel_size": 3,
        }
    ).to(device=device)

    torch.cuda.synchronize()
    start_time = time.time()
    summary(model, input_size=(8, 3, 512, 512), device=device, verbose=args.verbose)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Segmentation runtime: {end_time-start_time:.4f}s")

    print("\n\n\n" + "".join(["#"] * 95) + "\n\n\n")

    print("Verify SegOne Segmentation Architecture")
    model = SegOne(
        {
            "type": "segmentation",
            "channel_in": 3,
            "channel_out": 13,
            "num_layers": 4,
            "bottleneck_scale": 2,
            "bottleneck_repeat": 3,
            "bottleneck_channel": 32,
            "kernel_size": 3,
        }
    ).to(device=device)

    torch.cuda.synchronize()
    start_time = time.time()
    summary(model, input_size=(8, 3, 512, 512), device=device, verbose=args.verbose)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Segmentation runtime: {end_time-start_time:.4f}s")

    print("\n\n\n" + "".join(["#"] * 95) + "\n\n\n")

    print("Verify ResNet Segmentation Architecture")
    model = CommonNet(
        {
            "name": "RESNET",
            "type": "segmentation",
            "channel_in": 3,
            "channel_out": 38,
            "num_layers": 34,
            "bottleneck_scale": 2,
            "bottleneck_repeat": 3,
            "bottleneck_channel": 32,
            "kernel_size": 3,
        }
    ).to(device=device)

    torch.cuda.synchronize()
    start_time = time.time()
    summary(model, input_size=(8, 3, 512, 512), device=device, verbose=args.verbose)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Segmentation runtime: {end_time-start_time:.4f}s")
