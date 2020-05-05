#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" ResNet Agent """

import pandas as pd
import numpy as np
import pytorch_lightning as ptl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torchvision import transforms
from typing import Union, List
from lab.networks import ResNet


class Agent(ptl.LightningModule):
    def __init__(self, hparams, device=None, trial=None):
        """ Initialize Agent

        Args:
            hparams: Hyperparameters, passed as NestedSpace.
            device: torch device for this module.
            trial: Optuna trial object.
        """
        super(Agent, self).__init__()

        # Universal
        self.device = device
        self.hparams = hparams
        # ---------------------

        # Define network(s)
        self.net = ResNet(
            size=self.hparams.resnet_size,
            num_classes=10)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """ Set training

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # Pull Batch Info
        x, y_true = batch
        # Get model prediction
        y_hat = self.forward(x)
        # Cross entropy for train loss
        train_loss = F.cross_entropy(y_hat, y_true)
        # Any reporting logic
        with torch.no_grad():
            logs = {'train_loss': train_loss.item(),
                    'epoch': self.current_epoch}
        return {'loss': train_loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        """ set validation

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # Pull Batch Info
        x, y_true = batch
        # Get model prediction
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y_true)
        val_acc = (torch.softmax(y_hat, dim=1).argmax(dim=1) == y_true).to(torch.float32).mean()

        return {'val_loss': val_loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        """

        Args:
            outputs:

        Returns:

        """
        logs = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in outputs[0].keys()}
        logs['epoch'] = self.current_epoch
        return {'progress_bar': logs, 'log': logs}

    def test_step(self, batch, batch_idx):
        """ on test data

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # Pull Batch Info
        x, y_true = batch
        # Get model prediction
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y_true)
        test_acc = (torch.softmax(y_hat, dim=1).argmax(dim=1) == y_true).to(torch.float32).mean()
        return {'test_loss': test_loss, 'test_acc': test_acc}

    def test_epoch_end(self, outputs):
        """

        Args:
            outputs:

        Returns:

        """
        logs = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in outputs[0].keys()}
        return {'progress_bar': logs, 'log': logs}

    def _get_train_val_dataset(self):
        """CIFAR10 train and val datasets

        Returns:

        """
        dataset = CIFAR10(root='~/datasets/cifar10',
                          train=True, download=True,
                          transform=transforms.ToTensor())  # if self._train_dataset is None else self._train_dataset
        # random split save as self._train
        # Create train and validation datasets
        val_size = int(self.hparams.val_split * len(dataset))
        self._train_dataset, self._val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    # def test_step(self, batch, batch_idx):
    #     # Pull Batch Info
    #     x, y_true = batch
    #     # Get model prediction
    #     y_hat = self.forward(x)
    #     test_loss = F.cross_entropy(y_hat, y_true)
    #     test_acc = (torch.softmax(y_hat, dim=1).argmax(dim=1) == y_true).to(torch.float32).mean()
    #     # self._draw_labeled_images(x, torch.softmax(y_hat, dim=1).argmax(dim=1) )
    #     return {'test_loss': test_loss, 'test_acc': test_acc}
    #
    # def test_epoch_end(self, outputs):
    #     logs = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in outputs[0].keys()}
    #     return {'progress_bar': logs, 'log': logs}

    def _draw_labeled_images(self, images, labels):
        # the following episode is to view one set of images via tensorboard.
        from torchvision.utils import make_grid
        from matplotlib import pyplot as plt
        from torch.utils.tensorboard import SummaryWriter
        import time

        plt.ion()
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self._get_train_val_dataset()
        return DataLoader(self._train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4,
                          )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self._val_dataset is None:
            self._get_train_val_dataset()
        return DataLoader(self._val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataset = CIFAR10(root='~/datasets/cifar10',
                          train=False, download=True,
                          transform=transforms.ToTensor())

        print(self.hparams)
        self.hparams.batch_size = 128

        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)


    @staticmethod
    def add_module_args(parser):
        """ Additional hparams for this LightningModule and default reassignments
        """
        # Additional hparams

        parser.add_argument('--shuffle_minibatches', default=1, type=int)

        parser.add_argument('--resnet_size', default=20, type=int,
                            choices=list(ResNet.size_map.keys()))

        parser.add_argument('--learning_rate', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)

        # Update defaults
        parser.set_defaults(nepochs=200, nminibatches=1,
                            batch_size=128, dataset='cifar10',
                            monitor='val_loss')

        return parser

    @classmethod
    def get_pbt_hparams(cls):
        return dict(
            optimizer_states=[('lr', 'weight_decay', 'momentum')])



    @staticmethod
    def setup_trial(hparams, trial):
        """ Defines Trial Args for this LightningModule
        """
        hparams.learning_rate = trial.suggest_loguniform('learning_rate', 1.e-3, 1.e-1)
        hparams.weight_decay = trial.suggest_loguniform('weight_decay', 1.e-4, 1.e-3)
        hparams.resnet_size = trial.suggest_categorical('resnet_size', list(ResNet.size_map.keys()))

        return hparams, trial
