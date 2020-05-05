#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """
import os
import lab
from typing import Dict

from lab import test
from lab import train
from lab import _logger as log
from lab.core import setup_run

__all__ = [
    'alg_keys',
    'run_test',
    'run_train',
]


def alg_keys(lk=None):
    lk = lk or {}
    # Correct base run dir
    root_dir = os.path.split(os.path.dirname(lab.__file__))[0]
    os.chdir(root_dir)
    agent_dir = os.path.join(root_dir, 'lab', 'ptl_agents')
    # filter dataset agents only
    agent_lk = {os.path.splitext(name)[0]: {} for name in os.listdir(agent_dir)
                if os.path.isfile(os.path.join(agent_dir, name))
                and os.path.splitext(name)[0].endswith('_ds')}
    agent_lk.update(lk)
    return agent_lk.keys()


def run_train(kwargs: Dict):
    alg = kwargs.get('alg', None)
    if alg is None:
        raise RuntimeError
    cmd_args = [''] + [f'--{k}={v}' for k, v in kwargs.items()]
    try:
        train.main(*setup_run(args=cmd_args, mode='train'))
    except Exception as e:
        log.exception(f'Encountered excteption for alg={alg}')
        raise e
    else:
        log.info(f'Completed train for alg={alg}')

def run_test(kwargs: Dict):
    alg = kwargs.get('alg', None)
    if alg is None:
        raise RuntimeError
    cmd_args = [''] + [f'--{k}={v}' for k, v in kwargs.items()]
    try:
        test.main(*setup_run(args=cmd_args, mode='test'))
    except Exception as e:
        log.exception(f'Encountered excteption for alg={alg}')
        raise e
    else:
        log.info(f'Completed train for alg={alg}')
