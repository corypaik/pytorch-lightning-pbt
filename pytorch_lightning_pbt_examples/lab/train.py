#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Training Script """

import sys
import os
import pytorch_lightning as ptl
from lab.core import get_dl
from lab.core import setup_run
# from pytorch_lightning_pbt import loggers
from pytorch_lightning import loggers

import pytorch_lightning_pbt as pbt


def main(alg_module, hparams, device):
    """ Standard Pytorch Lightning Runner.

    Args:
        alg_module: module containing an agent class, i.e `class Agent(ptl.LightningModule)`.
        hparams: hyperparameters for the agent, parsed from argument parser.
        device: torch device for the model.

    """

    # setup agent
    model = alg_module.Agent(hparams=hparams, device=device)
    #checkpoint_dir = os.path.join(hparams.parent_dir, 'checkpoints')

    # if hparams.use_pbt:
    #save_dir = os.path.join('data', hparams.alg, hparams.dataset)
    logger = loggers.TensorBoardLogger(save_dir='data', name=hparams.alg, version=hparams.version)

    trainer = ptl.Trainer(
        logger=True,
        checkpoint_callback=True,
        default_root_dir='data',
        max_epochs=hparams.nepochs,
        min_epochs=hparams.nepochs,
        gpus=None if str(device) == 'cpu' else 1,
        gradient_clip_val=hparams.max_grad_norm,
        log_gpu_memory=True,
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=hparams.val_interval,
        reload_dataloaders_every_epoch=False,
        track_grad_norm=2,
        process_position=1,
    )

    if hparams.use_pbt:
        trainer = pbt.Trainer(
            trainer=trainer,
            # New Args.
            population_size=hparams.population_size,
            num_workers=hparams.num_workers,
            # optional.
            tune_hparams=None,
            pbt_monitor=hparams.monitor,
            pbt_period=None,
        )

    # get dataloaders and run
    trainer.fit(model=model, **get_dl(hparams=hparams, ds_types='train_val'))
    # trainer.test(model, **get_dl(hparams=hparams, ds_types='test'))


if __name__ == '__main__':
    main(*setup_run(args=sys.argv, mode='train'))
