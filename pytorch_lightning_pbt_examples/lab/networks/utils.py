#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'loguniform',
    'Flatten',
    'View',
]


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


# noinspection PyMethodMayBeStatic
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


class View(nn.Module):
    def __init__(self, shape: Tuple):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(*self.shape)
