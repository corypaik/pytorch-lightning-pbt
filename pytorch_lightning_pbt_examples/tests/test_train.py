#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """

# Local
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
    num_workers=1,
    population_size=1,
)

@pytest.mark.agents
@pytest.mark.parametrize("alg", tutils.alg_keys())
def test_pl_train_defaults(alg):
    """ Tests

    Args:
        alg:

    Returns:

    """
    kwargs = common_kwargs.copy()
    kwargs['alg'] = alg
    tutils.run_train(kwargs=kwargs)


@pytest.mark.parametrize("alg", tutils.alg_keys())
def test_pbt_train_defaults(alg):
    """ Tests

    Args:
        alg:

    Returns:

    """
    kwargs = common_kwargs.copy()
    kwargs['alg'] = alg
    kwargs['use_pbt'] = 1
    tutils.run_train(kwargs=kwargs)