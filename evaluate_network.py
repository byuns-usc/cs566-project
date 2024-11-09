import argparse
import os

import torch
from ruamel.yaml import YAML

from datasets.dataloaders import create_dataloader
from segone.utils.eval import Evaluator

# Predicted one-hot encoded tensor (N, C, H, W)
# Ground truth label tensor (N, H, W) with values from 0 to C-1

available_datasets = ("COCO", "VOC", "PET", "PET2", "BRAIN", "HEART")
available_models = {
    "SEGONE": SegOne,
    "ONENET": SegOne,
    "RESNET": CommonNet,
    "UNET": CommonNet,
    "MOBILENET": CommonNet,
    # "SKIPINIT": CommonNet,
    # "EUNNET": CommonNet,
}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=True)
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()

# Load options
with open(args.cfg) as cfg_file:
    yaml = YAML(typ="safe")
    opts = yaml.load(cfg_file)
data_opts = opts["data"]
train_opts = opts["train"]
model_opts = opts["model"]

# Set device
device = torch.device(f"cuda:{args.cuda}" if train_opts["cuda"] and torch.cuda.is_available() else "cpu")

# Load Data
assert data_opts["name"] in available_datasets
assert data_opts["resolution"][0] % 32 == 0 and data_opts["resolution"][1] % 32 == 0
train_loader = create_dataloader(
    data_opts["datapath"],
    data_opts["name"],
    split=data_opts["train"],
    batch_size=train_opts["batch_size"],
    img_size=data_opts["resolution"],
    num_workers=train_opts["num_workers"],
)
val_loader = create_dataloader(
    data_opts["datapath"],
    data_opts["name"],
    split=data_opts["val"],
    batch_size=train_opts["batch_size"],
    img_size=data_opts["resolution"],
    num_workers=train_opts["num_workers"],
)

assert model_opts["name"] in available_models
model = available_models[model_opts["name"]](model_opts)
model.to(device)

if not model_opts["name"] == "RESNET":
    print("Initializing Weight")
    initialize_weights()

# If pretrain model exists, load
if train_opts["load_weights"] is not None:
    load_model()

try:
    model.load_state_dict(torch.load(train_opts["load_weights"], weights_only=True))
except:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(train_opts["load_weights"], weights_only=True)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

evaluator = Evaluator(preds, target, num_classes=C)
results = evaluator.evaluate_all()

for metric, value in results.items():
    print(f"{metric}: {value}")
