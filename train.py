import argparse
import os

import torch
from ruamel.yaml import YAML

from segone.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    # Load options
    with open(args.cfg) as cfg_file:
        yaml = YAML(typ="safe")
        opts = yaml.load(cfg_file)

    trainer = Trainer(opts, args.cuda)
    trainer.train()
    torch.cuda.empty_cache()