#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Main Testing Script """

import os
import sys

from lab import _logger as log
from lab.core import get_dl
from lab.core import setup_run


def main(alg_module, hparams, device):
    """ Main ptl test runner

    Args:
        alg_module: module containing an agent class, i.e `class Agent(ptl.LightningModule)`.
        hparams: hyperparameters for the agent, parsed from argument parser.
        device: torch device for the model.

    """

    checkpoint_dir = os.path.join(hparams.parent_dir, 'checkpoints')

    ckpt_paths = [os.path.join(checkpoint_dir, ckpt_name) for ckpt_name in os.listdir(checkpoint_dir)]
    if len(ckpt_paths) == 0:
        raise logger.exception(f'No checkpoints found in {checkpoint_dir}')

    # load and freeze agent
    agent = alg_module.Agent.load_from_checkpoint(checkpoint_path=ckpt_paths[0], device=device)
    agent.freeze()

    # Test dataloader
    dataloaders = get_dl(hparams=hparams, ds_types='test')

    # Test testing
    output = agent.test_epoch_end(
        outputs=[agent.test_step(batch=batch, batch_idx=i)
                 for i, batch in enumerate(dataloaders['test_dataloaders'])]).get('log', {})
    log.info(output)

if __name__ == '__main__':
    main(*setup_run(args=sys.argv, mode='test'))
