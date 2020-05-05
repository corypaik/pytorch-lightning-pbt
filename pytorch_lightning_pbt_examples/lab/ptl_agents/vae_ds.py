#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Variational Autoencoder (VAE) for Datasets.

    Compatibility:
        - MNIST
"""

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lab.networks.functional import *
from lab.networks.utils import *
from torchvision.utils import make_grid


class VAE(torch.nn.Module):
    def __init__(self, hparams, z_dim=62, batch_norm=True):
        """

        Args:
            hparams: Hyperparameters, passed as NestedSpace.
            z_dim:
            batch_norm: True
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim
        # Set for dataset
        if hparams.dataset == 'mnist':
            w, h, ic = (28, 28, 1)
        elif hparams.dataset.startswith('cifar'):
            w, h, ic = (32, 32, 3)
        # Define encoder
        self.encoder = nn.Sequential(
            conv(ic=ic, oc=64, ks=4, s=2, pad='valid', padding=1, gain=np.sqrt(2)), nn.LeakyReLU(0.2),
            conv(ic=64, oc=128, ks=4, s=2, pad='valid', padding=1, gain=np.sqrt(2)),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2), Flatten(),
            fc(id=128 * (w // 4) * (h // 4), od=1024, gain=np.sqrt(0.1)), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            fc(id=1024, od=2 * self.z_dim))
        # Define decoder
        self.decoder = nn.Sequential(
            fc(id=self.z_dim, od=1024), nn.BatchNorm1d(1024), nn.ReLU(),
            fc(id=1024, od=128 * (w // 4) * (h // 4), gain=np.sqrt(0.1)),
            nn.BatchNorm1d(128 * (w // 4) * (h // 4)), nn.ReLU(), View(shape=(-1, 128, (w // 4),  (h // 4))),
            convt(ic=128, oc=64, ks=4, s=2, pad='valid', padding=1, ), nn.BatchNorm2d(64), nn.ReLU(),
            convt(ic=64, oc=ic, ks=4, s=2, pad='valid', padding=1, ), nn.Sigmoid())

    def forward(self, img):
        """

        Args:
            img: encoder(img)

        Returns:

        """
        enc_ret = self.encoder(img)
        mu, log_sigma = torch.split(enc_ret, split_size_or_sections=enc_ret.size(1) // 2, dim=1)
        z = self.reparameterize(mu, log_sigma)
        return self.decoder(z), mu, log_sigma

    @staticmethod
    def reparameterize(mu, log_sigma):
        """

        Args:
            mu:
            log_sigma:

        Returns:

        """
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    @staticmethod
    def _initialize_weights(m):
        """

        Args:
            m:

        Returns:

        """
        # Check and set weights and bias
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Agent(ptl.LightningModule):
    def __init__(self, *, hparams, device=None):
        """ Initialize Agent

        Args:
            venv: OpenAI VecEnv or CuLE (Only here for compatibility)
            hparams: Hyperparameters, passed as NestedSpace
            device: torch device for this module
            trial: optuna trial (Compatibility)
        """
        super(Agent, self).__init__()

        # Universal
        self.device = device
        self.hparams = hparams

        # Define network(s)
        self.net = VAE(hparams=hparams)

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        return self.net(x)

    def configure_optimizers(self):
        """

        Returns:

        """
        return [torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate, eps=1.e-8)]

    @staticmethod
    def vae_loss(recon_x, x, mu, log_sigma):
        """calculate vae loss

        Args:
            recon_x:
            x:
            mu:
            log_sigma:

        Returns: tot_loss, recon_loss, kl_loss

        """
        recon_x = recon_x.to(torch.float32)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1.0 - log_sigma, 1))
        tot_loss = recon_loss + kl_loss
        return tot_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """ set training

        Args:
            batch:
            batch_idx:

        Returns: train_loss, logs

        """
        # Pull Batch Info
        x, y = batch
        recon_x, mu, logvar = self.net(x)
        # obtain respective loss
        train_loss, recon_loss, kl_loss = self.vae_loss(recon_x, x, mu, logvar)
        # Any reporting logic
        with torch.no_grad():
            logs = {'train_loss': train_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item()}
        return {'loss': train_loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        """ set validation

        Args:
            batch:
            batch_idx:

        Returns:loss, recon_loss, kl_loss

        """
        # Pull Batch Info
        x, y = batch
        recon_x, mu, logvar = self.net(x)
        # obtain respective loss
        loss, recon_loss, kl_loss = self.vae_loss(recon_x, x, mu, logvar)
        # Any reporting logic
        with torch.no_grad():
            # write random sample of images
            if self.current_epoch % self.hparams.log_images_interval == batch_idx == 0:
                self.log_images(x, recon_x)
        return {'val_loss': loss, 'val_recon_loss': recon_loss, 'val_kl_loss': kl_loss}

    def test_step(self, batch, batch_idx):
        """ set validation

        Args:
            batch:
            batch_idx:

        Returns:loss, recon_loss, kl_loss

        """
        # Pull Batch Info
        x, y = batch
        recon_x, mu, logvar = self.net(x)
        # obtain respective loss
        loss, recon_loss, kl_loss = self.vae_loss(recon_x, x, mu, logvar)
        # Any reporting logic
        with torch.no_grad():
            # write random sample of images
            if self.current_epoch % self.hparams.log_images_interval == batch_idx == 0:
                self.log_images(x, recon_x)
        return {'test_loss': loss, 'test_recon_loss': recon_loss, 'test_kl_loss': kl_loss}

    def test_epoch_end(self, outputs):
        """

        Args:
            outputs:

        Returns: logs

        """
        logs = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in outputs[0].keys()}
        logs['epoch'] = self.current_epoch
        return {'progress_bar': logs, 'log': logs}

    def validation_epoch_end(self, outputs):
        """

        Args:
            outputs:

        Returns: logs

        """
        logs = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in outputs[0].keys()}
        logs['epoch'] = self.current_epoch
        return {'progress_bar': logs, 'log': logs}

    def log_images(self, x, recon_x):
        """ Log grid of images to TensorBoard

        Args:
            x:
            recon_x:

        Returns:

        """

        grid_x = make_grid(x[:self.hparams.grid_size], nrow=2)
        self.logger.experiment.add_image('x_true', grid_x, global_step=self.current_epoch)

        grid_recon_x = make_grid(recon_x[:self.hparams.grid_size], nrow=2)
        self.logger.experiment.add_image('recon_x', grid_recon_x, global_step=self.current_epoch)

    @classmethod
    def get_pbt_hparams(cls):
        """Initialize hyperparameters wanted for optimization

        Returns:

        """
        return dict(
            optimizer_states=[('lr', )])

    @staticmethod
    def add_module_args(parser):
        """ Additional hparams for this LightningModule and default reassignments
        """
        # Additional hparams
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--log_images_interval', type=int, default=10)
        parser.add_argument('--grid_size', type=int, default=9)

        # Update defaults
        parser.set_defaults(nepochs=300, nminibatches=4, dataset='mnist',
                            monitor='val_loss', reload_dataloaders_every_epoch=0,
                            dl_num_workers=2, num_workers=8, population_size=16)

        return parser
