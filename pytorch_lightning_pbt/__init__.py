#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """
import logging as python_logging
_logger = python_logging.getLogger('lightning:pbt')
python_logging.basicConfig(level=python_logging.INFO)

# Code bade info
__version__ = '0.0.1'
__author__ = 'Cory Paik'
__author_email__ = 'corypaik@gmail.com'
__homepage__ = 'https://github.com/coypaik/pytorch-lightning-pbt'
__docs__ = "PyTorch Lightning Population Based Training."

from pytorch_lightning_pbt.trainer import Trainer

__all__ = [
    'Trainer',
]
