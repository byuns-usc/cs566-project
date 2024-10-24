import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from ruamel.yaml import YAML
from tqdm import tqdm

from datasets.dataloaders import create_dataloader
from segone.networks.common_network import CommonNet
from segone.networks.segone_network import SegOne


class Trainer:
    available_datasets = ("COCO", "VOC", "PET", "BRAIN", "HEART")
    available_models = {
        "SEGONE": SegOne,
        "RESNET": CommonNet,
        "UNET": CommonNet,
        "SKIPINIT": CommonNet,
        "EUNNET": CommonNet,
    }

    def __init__(self, opts):
        self.data_opts = opts["data"]
        self.train_opts = opts["train"]
        self.model_opts = opts["model"]

        self.device = torch.device("cuda" if self.train_opts["cuda"] and torch.cuda.is_available() else "cpu")

        # Load Data
        assert self.data_opts["name"] in self.available_datasets
        assert self.data_opts["resolution"][0] % 32 == 0 and self.data_opts["resolution"][1] % 32 == 0
        self.train_loader = create_dataloader(
            self.data_opts["datapath"],
            self.data_opts["name"],
            split=self.data_opts["train"],
            batch_size=self.train_opts["batch_size"],
            img_size=self.data_opts["resolution"],
            num_workers=self.train_opts["num_workers"],
        )
        self.val_loader = create_dataloader(
            self.data_opts["datapath"],
            self.data_opts["name"],
            split=self.data_opts["val"],
            batch_size=self.train_opts["batch_size"],
            img_size=self.data_opts["resolution"],
            num_workers=self.train_opts["num_workers"],
        )

        self.val_iter = iter(self.val_loader)
        inputs = next(self.val_iter)
        print(len(inputs))
        print(inputs[0].size())

        # Define model
        assert self.model_opts["name"] in self.available_models
        self.model = self.available_models[self.model_opts["name"]](self.model_opts)
        self.model.to(self.device)

        # If pretrain model exists, load
        if self.train_opts["load_weights"] is not None:
            self.load_model()
        elif self.train_opts["load_encoder"] is not None:
            self.load_encoder()

        # Set training params
        self.optim = optim.Adam(self.model.parameters(), self.train_opts["learning_rate"])
        self.criteria = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, self.train_opts["lr_step"], 0.1)

        # Make dir and save options used
        self.save_dir = os.path.join(self.train_opts["save_dir"], str(int(time.time())))
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "config.yaml"), "wb") as f:
            yaml = YAML()
            yaml.dump(opts, f)

    def process_batch(self, inputs):
        images, targets = inputs
        outputs = self.model(images)
        losses = self.criteria(outputs, targets)

        outputs, losses

    def train(self):
        self.model.train()
        for self.epoch in range(self.train_opts["epoch"]):
            self.lr_scheduler.step()

            for batch_idx, inputs in tqdm(enumerate(self.train_loader)):
                outputs, losses = self.process_batch(inputs)
                self.model_optimizer.zero_grad()
                losses.backward()
                self.model_optimizer.step()

            val_losses = self.val()

            print(f"Train loss: {losses}, Val loss: {val_losses}")

            if (self.epoch + 1) % self.train_opts["save_frequency"] == 0:
                self.save_model()

    def val(self):
        self.model.eval()

        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            del inputs, outputs

        self.model.train()
        return losses

    def save_model(self):
        weight_save_dir = os.path.join(self.save_dir, "model")
        weight_save_path = os.path.join(weight_save_dir, "weights_{}.pth".format(self.epoch))

        if not os.path.exists(weight_save_dir):
            os.makedirs(weight_save_dir)
        torch.save(self.model.state_dict(), weight_save_path)

        optim_save_path = os.path.join(weight_save_dir, "adam_.pth".format(self.epoch))
        torch.save(self.optim.state_dict(), optim_save_path)

    def load_model(self):
        # Load model weights
        print(f"Loading weights at {self.train_opts["load_weights"]}")
        try:
            self.model.load_state_dict(torch.load(self.train_opts["load_weights"]))
        except:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.train_opts["load_weights"])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # Load optimizer
        if self.train_opts["load_optimizer"] is not None:
            print(f"Loading optimizer at {self.train_opts["load_optimizer"]}")
            self.optim.load_state_dict(torch.load(self.train_opts["load_optimizer"]))
        else:
            print("Warning: loading weights without optimizer")

    def load_encoder(self):
        print(f"Loading encoder weights at {self.train_opts["load_encoder"]}")
        try:
            self.model.encoder.load_state_dict(torch.load(self.train_opts["load_encoder"]))
        except:
            encoder_dict = self.model.encoder.state_dict()
            pretrained_dict = torch.load(self.train_opts["load_encoder"])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            encoder_dict.update(pretrained_dict)
            self.model.encoder.load_state_dict(encoder_dict)
