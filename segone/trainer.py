import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ruamel.yaml import YAML
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.dataloaders import create_dataloader
from segone.networks.one_encoder import OneEncoder
from segone.networks.seg_decoder import SegDecoder
from segone.utils.layers import calculate_losses

class Trainer:
    available_datasets = ("COCO", "VOC", "PET", "BRAIN", "HEART")

    def __init__(self, opts):
        self.data_opts = opts["data"]
        self.train_opts = opts["train"]
        self.model_opts = opts["model"]

        self.device = torch.device("cuda" if self.train_opts.cuda and torch.cuda.is_available() else "cpu")

        # TODO Load Data
        assert self.data_opts["name"] in self.available_datasets
        self.train_dataset
        self.val_dataset
        self.train_loader
        self.val_loader

        # TODO Define model
        model = None
        if self.train_opts["load_weights"] is not None:
            self.load_model()

        # Set training params
        self.optim = optim.Adam(self.model.parameters(), self.train_opts["learning_rate"])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, self.train_opts["lr_step"], 0.1)

        # Make dir and save options used
        self.save_dir = os.path.join(self.train_opts["save_dir"], str(int(time.time())))
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "config.yaml"), "wb") as f:
            yaml = YAML()
            yaml.dump(opts, f)

    def train(self):
        self.model.train()

        for self.epoch in range(self.opt.num_epochs):
            self.model_lr_scheduler.step()

            for batch_idx, inputs in enumerate(self.train_loader):
                outputs, losses = self.process_batch(inputs)

                self.model_optimizer.zero_grad()
                losses.backward()
                self.model_optimizer.step()

            self.val()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
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
            del inputs, outputs, losses

        self.model.train()

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}

        return losses

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == "encoder":
                # save the sizes - these are needed at prediction time
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
                to_save["use_stereo"] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
