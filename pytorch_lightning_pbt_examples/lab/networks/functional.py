#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: Apr. 2020
# ---------------------
""" PyTorch Custom Layers and Layers w/ initialization (semi-functional) """

import numpy as np
import torch
import torch.nn as nn


__all__ = [
    'fc',
    'conv',
    'convt',
]


def convt(*, ic, oc, ks, s, pad='zeros', gain=1., c=0.0, dims='2d', padding=0, trainable=True,
          ki='orthogonal'):
    if pad == 'valid':
        from pytorch_lightning_pbt_examples.lab import _logger
        _logger.warn(f'padding_mode=valid depreciated in torch 1.5.0.'
                     f'changing to padding_mode=zeros for compatibility.')
        pad = 'zeros'

    layer = nn.ConvTranspose2d(in_channels=ic, out_channels=oc, kernel_size=ks,
                               stride=s, padding_mode=pad, padding=padding)

    # Initialize
    if ki == 'orthogonal':
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
    elif ki == 'golort_uniform':
        torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    else:
        raise NotImplementedError

    torch.nn.init.constant_(layer.bias, val=c)

    if not trainable:
        for param in layer.parameters():
            param.requires_grad = False

    return layer


def conv(*, ic, oc, ks, s, pad='zeros', gain=1., c=0.0, dims='2d', padding=0, trainable=True,
         ki='orthogonal'):
    if pad == 'valid':
        from pytorch_lightning_pbt_examples.lab import _logger
        _logger.warn(f'padding_mode=valid depreciated in torch 1.5.0.'
                     f'changing to padding_mode=zeros for compatibility.')
        pad = 'zeros'

    layer = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=s, padding_mode=pad, padding=padding)

    # Initialize
    if ki == 'orthogonal':
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
    elif ki == 'golort_uniform':
        torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    else:
        raise NotImplementedError

    torch.nn.init.constant_(layer.bias, val=c)

    if not trainable:
        for param in layer.parameters():
            param.requires_grad = False

    return layer


def fc(*, id, od, gain=np.sqrt(0.01), c=0.0, trainable=True,
       ki='orthogonal'):
    layer = nn.Linear(in_features=id, out_features=od)
    # Initialize
    if ki == 'orthogonal':
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
    elif ki == 'normc':
        raise NotImplementedError
    else:
        raise NotImplementedError

    torch.nn.init.constant_(layer.bias, val=c)

    if not trainable:
        for param in layer.parameters():
            param.requires_grad = False

    return layer
