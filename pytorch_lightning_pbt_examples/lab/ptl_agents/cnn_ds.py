#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Baseline Agent:

Sequential(
  (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2))
  (1): ReLU()
  (2): Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1))
  (3): ReLU()
  (4): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
  (5): ReLU()
  (6): nn_flatten()
  (7): Linear(in_features=10816, out_features=43, bias=True)
)

"""

from argparse import Namespace
from typing import Union, List

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lab.networks.utils import Flatten
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Agent(ptl.LightningModule):

    def __init__(self, hparams: Namespace, device=None):
        """ Initialize Agent

        Args:
            hparams: Hyperparameters, passed as Namespace.
            device: torch device for this module.
        """
        super(Agent, self).__init__()

        # Universal
        self.device = device
        self.hparams = hparams
        # ---------------------

        # Define network(s)
        ic = 3 if hparams.dataset.startswith('cifar') else 1
        fft = 5184 if hparams.dataset.startswith('cifar') else 0
        self.net = nn.Sequential(
            nn.Conv2d(ic, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 1), nn.ReLU(), Flatten(),
            nn.Linear(fft, 10))

        # initialize
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight, gain=1.)
            torch.nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(0.01))
            torch.nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        """ pass in net(cnn)

        Args:
            x:

        Returns:

        """
        return self.net(x)

    def configure_optimizers(self):
        """ return configure optimizer

        Returns:

        """
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, eps=1.e-8)]

    def training_step(self, batch, batch_idx):
        """set training

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
        """set validation

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
        """on test data

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

    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     """set CIFAR10 for test dataloader
    #
    #     Returns:
    #
    #     """
    #     dataset = CIFAR10(root='~/datasets/cifar10',
    #                       train=False, download=True,
    #                       transform=transforms.ToTensor())
    #     return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    @classmethod
    def get_pbt_hparams(cls):
        """ Initialize hyperparameters wanted for optimization

        Returns:

        """
        return dict(
            optimizer_states=[('lr', )])

    @staticmethod
    def add_module_args(parser):
        """ Additional hparams for this LightningModule and default reassignments
        """
        parser.add_argument('--learning_rate', default=1.e-4, type=float)
        parser.add_argument('--shuffle_minibatches', default=1, type=int)

        # Update defaults
        parser.set_defaults(nepochs=300, nminibatches=1, dataset='cifar10', monitor='val_loss',
                            batch_size=512, val_split=0.2)

        return parser
