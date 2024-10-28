import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ruamel.yaml import YAML

from datasets.dataloaders import create_dataloader
from segone.networks.common_network import CommonNet
from segone.networks.segone_network import SegOne


def load_model(model, weight_path):
    print(f"Loading weights at {weight_path}")
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weight_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


def plot_mask(images, targets, masks):
    """Given tensors, save as plotted images"""
    images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)
    targets = targets.cpu().detach().numpy()
    masks = masks.cpu().detach()
    masks = torch.argmax(masks, dim=1).numpy()

    fig, axs = plt.subplots(2, 6)
    cmap = plt.get_cmap("viridis", 38)
    for i in range(2):
        for j in range(2):

            # print(np.unique(targets[i*2+j], return_counts=True))
            # print(np.unique(masks[i*2+j], return_counts=True))

            axs[i, j * 3].imshow(images[i * 2 + j])
            axs[i, j * 3 + 1].imshow(targets[i * 2 + j], cmap=cmap, vmin=0, vmax=38)
            axs[i, j * 3 + 2].imshow(masks[i * 2 + j], cmap=cmap, vmin=0, vmax=38)
            axs[i, j * 3].axis("off")
            axs[i, j * 3 + 1].axis("off")
            axs[i, j * 3 + 2].axis("off")

            axs[i, j * 3].set_xticks([])
            axs[i, j * 3].set_yticks([])
            axs[i, j * 3 + 1].set_xticks([])
            axs[i, j * 3 + 1].set_yticks([])
            axs[i, j * 3 + 2].set_xticks([])
            axs[i, j * 3 + 2].set_yticks([])
    plt.show()


if __name__ == "__main__":
    available_models = {
        "SEGONE": SegOne,
        "RESNET": CommonNet,
        "UNET": CommonNet,
        "SKIPINIT": CommonNet,
        "EUNNET": CommonNet,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    with open(args.cfg) as cfg_file:
        yaml = YAML(typ="safe")
        opts = yaml.load(cfg_file)
    model_opts = opts["model"]
    train_opts = opts["train"]
    data_opts = opts["data"]

    device = f"cuda:{args.cuda}" if args.cuda > -1 else "cpu"

    model = available_models[model_opts["name"]](model_opts)
    model.to(device)
    load_model(model, args.weight)

    loader = create_dataloader(
        data_opts["datapath"],
        data_opts["name"],
        split=data_opts[args.split],
        batch_size=train_opts["batch_size"],
        img_size=data_opts["resolution"],
        num_workers=train_opts["num_workers"],
    )

    loader_iter = iter(loader)
    images, targets = next(loader_iter)

    images = images.to(device)
    targets = targets.to(device)

    outputs = model(images)

    plot_mask(images, targets, outputs[-1])
