import argparse
import os

import matplotlib.pyplot
import torch
from ruamel.yaml import YAML

from segone.networks.common_network import CommonNet
from segone.networks.segone_network import SegOne


def load_model(model, weight_path):
    print(f"Loading weights at {weight_path}")
    try:
        model.load_state_dict(torch.load(weight_path))
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weight_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


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
    args = parser.parse_args()

    with open(args.cfg) as cfg_file:
        yaml = YAML(typ="safe")
        opts = yaml.load(cfg_file)
    model_opts = opts["model"]
    train_opts = opts["train"]

    device = f"cuda:{args.cuda}" if args.cuda > -1 else "cpu"

    assert model_opts["name"] in available_models
    assert train_opts["load_weights"] is not None

    model = available_models[model_opts["name"]](model_opts)
    model.to(device)
    load_model(args.weight)
