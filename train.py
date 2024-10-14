import os
import argparse

from ruamel.yaml import YAML
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()

    # Load options
    with open(args.cfg) as cfg_file:
        yaml = YAML(typ='safe')
        opts = yaml.load(cfg_file)

    trainer = Trainer(opts)
    trainer.train()
