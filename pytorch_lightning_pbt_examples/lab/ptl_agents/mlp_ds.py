#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Basic Multilayer Perceptron (MLP) for Datasets

    Compatibility:
        MNIST
"""

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(ptl.LightningModule):
    def __init__(self, hparams, device=None):
        """ Initialize Agent

        Args:
            hparams: Hyperparameters, passed as NestedSpace
            device: torch device for this module
        """
        super(Agent, self).__init__()

        self.device = device
        self.hparams = hparams

        # Network(s)
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10))

        self._pbt_task_id = -1

    def forward(self, x):
        """pass in net

        Args:
            x:

        Returns:

        """
        x = x.view(-1, 784)
        return self.net(x)

    def configure_optimizers(self):
        """ Slight modifications to this function for PBT.

        Examples:
            ```python
                # Without Population Based Training.
                optimizers = [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, eps=1.e-8)]

                # With Population Based Training.
                # first sample initial hypereparameters.
                lr = np.random.choice(np.logspace(-5, 0, base=10))
                momentum = np.random.choice(np.linspace(0.1, .9999))
                # return is exactly the same.
                return [torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)]
            ```
        Returns:

        """
        # With Population Based Training.
        # first sample initial hypereparameters.
        if self.hparams.use_pbt:
            lr = np.random.choice(np.logspace(-5, 0, base=10))
            momentum = np.random.choice(np.linspace(0.1, .9999))
        else:
            lr = self.hparams.learning_rate
            momentum = 0.9
        # return is exactly the same.
        return [torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)]

    def training_step(self, batch, batch_idx):
        """Set training vals

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # Pull Batch Info
        images, targets = batch
        # Get Model prediction
        y_hat = self.forward(images)
        # Cross entropy for train loss
        train_loss = F.cross_entropy(y_hat, targets)
        return {'loss': train_loss, 'log': {'train_loss': train_loss.item(), 'task_id': self._pbt_task_id}}

    def validation_step(self, batch, batch_idx):
        """ set validation

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # pull batch info
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
        logs['task_id'] = self._pbt_task_id
        return {'log': logs, **logs}

    def test_step(self, batch, batch_idx):
        """ set validation

        Args:
            batch:
            batch_idx:

        Returns:

        """
        # pull batch info
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
        logs['task_id'] = self._pbt_task_id
        return {'log': logs, **logs}

    @classmethod
    def get_pbt_hparams(cls):
        """ Initialize hyperparameters wanted for optimization

        Returns:

        """
        return dict(
            hparams=('batch_size',),
            optimizer_states=[('lr', 'momentum')])

    @staticmethod
    def add_module_args(parser):
        """ Additional hparams for this LightningModule and default reassignments
        """

        parser.add_argument('--learning_rate', default=1.e-3, type=float)
        parser.add_argument('--shuffle_minibatches', default=1, type=int)

        # Update defaults
        parser.set_defaults(nupdates=300, reload_dataloaders_every_epoch=0,
                            monitor='val_acc',
                            dataset='mnist', nepochs=100)

        return parser
