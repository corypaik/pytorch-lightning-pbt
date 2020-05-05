#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  TempusLab
#  Authors: Cory Paik
#  Updated: Apr. 2020
# ---------------------

import os

# Local
import lab
import pytest
from lab import _logger as log
from lab import train
from lab.core import setup_run


def get_algs(lk=None):
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
    return agent_lk


common_kwargs = dict(
    dataset='mnist',
    nupdates=1,
    nepochs=1,
    pytest=1,
    progressbar=0,
)

learn_kwargs = get_algs(lk={})

@pytest.mark.agents
@pytest.mark.parametrize("alg", learn_kwargs.keys())
@pytest.mark.skip('Not all mnist ')
def test_mnist(alg):
    # logger = get_logger('lab.core')
    kwargs = common_kwargs.copy()
    kwargs.update(learn_kwargs[alg])
    kwargs['alg'] = alg
    # make list
    cmd_args = [''] + [f'--{k}={v}' for k, v in kwargs.items()]
    try:
        train.main(*setup_run(args=cmd_args, mode='train'))
    except Exception as e:
        log.exception(f'Encountered excteption for alg={alg}')
        raise e
    else:
        log.info(f'Completed train for alg={alg}')






if __name__ == '__main__':
    get_algs({})
