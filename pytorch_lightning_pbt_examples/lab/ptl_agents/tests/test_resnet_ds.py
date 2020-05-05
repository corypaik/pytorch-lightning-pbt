#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """
import pytest
from lab.core.tests import utils as tutils

from lab import _logger as log
from lab import train
from lab.core import setup_run



common_kwargs = dict(
    nupdates=1,
    nepochs=1,
    pytest=1,
    progressbar=0,
    num_workers=4,
    population_size=4
)
from lab.networks.resnet import ResNet



@pytest.mark.parametrize(["resnet_size", "use_pbt"], [
    *[(size, 1) for size in ResNet.size_map.keys()],
    *[(size, 0) for size in ResNet.size_map.keys()],
])
def test_train_resnet_ds(resnet_size, use_pbt):
    """ Tests

    Args:
        alg:

    Returns:

    """
    kwargs = common_kwargs.copy()
    kwargs['resnet_size'] = resnet_size
    kwargs['use_pbt'] = use_pbt
    kwargs['alg'] = 'resnet_ds'
    tutils.run_train(kwargs=kwargs)

@pytest.mark.parametrize(["resnet_size", "use_pbt"], [
    *[(size, 1) for size in ResNet.size_map.keys()],
    *[(size, 0) for size in ResNet.size_map.keys()],
])
def test_test_resnet_ds(resnet_size, use_pbt):
    """ Tests

    Args:
        alg:

    Returns:

    """
    kwargs = common_kwargs.copy()
    kwargs['resnet_size'] = resnet_size
    kwargs['use_pbt'] = use_pbt
    kwargs['alg'] = 'resnet_ds'
    tutils.run_test(kwargs=kwargs)
