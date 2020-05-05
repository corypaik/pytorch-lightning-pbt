#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """


import copy
from typing import Dict, Optional

import pytorch_lightning as pl
import torch.multiprocessing as _mp
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from pytorch_lightning_pbt.core import Worker

class Trainer(object):

    def __init__(
            self,
            trainer: pl.Trainer,
            population_size: int = 1,
            num_workers: int = 4,
            tune_hparams: Optional[Dict] = None,
            pbt_monitor: Optional[str] = None,
            pbt_period: Optional[int] = None,
    ):
        """ Population Based Trainer

        Args:
            trainer (:class:`pytorch_lightning.Trainer`): Initialized trainer to copy across population
            population_size: Population size is the number of individual agent tasks. This must be >= num_workers.

                For optimal performance, population_size should be equal to the num_workers, in which case
                each member of the population is trained completely in parallel.

                If `population_size` is > `num_workers`, then they will be processed in a FIFO Queue. This is
                the preferred method for simulating larger populations on a single machine.

            num_workers: Number of parallel training processes.
            tune_hparams: dict containing parameters to tune for the module.
                If this is not passed, PBT will only be performing the Exploit part of the algorithm.
            pbt_monitor: Optional pbt_monitor override flag. If this is None, PBT will use the monitor flag
                from `trainer.checkpoint_callback.monitor`.
        """
        # change some args around.
        trainer.min_epochs = (trainer.min_epochs * population_size) // num_workers
        trainer.max_epochs = ((trainer.max_epochs * population_size) // num_workers) * 2
        # todo: Check args.

        # Monitor required either from `trainer.checkpoint_callback.monitor` or `pbt_monitor`

        if isinstance(trainer.checkpoint_callback, bool):
            if pbt_monitor is None:
                raise MisconfigurationException(
                    f'Must provide a monitor for PBT either through the pl.Trainer class, or by directly passing'
                    f'`pbt_parameter`. ')
        elif pbt_monitor is None:
            pbt_monitor = trainer.checkpoint_callback.monitor





        if trainer.checkpoint_callback is False and pbt_monitor is not None:
            pass # TODO.


        # store trainer
        self.trainer = trainer

        # pbt args.
        self.num_workers = num_workers
        self.population_size = population_size
        self.tune_hparams = tune_hparams
        self.pbt_monitor = pbt_monitor

        self.model = None

        if self.num_workers < self.population_size:
            pass

        # TODO: Verify PBT hparams.

    def fit(
            self,
            model: pl.LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[DataLoader] = None,
    ):
        self.model = model

        # try to get hparams from module.
        if self.tune_hparams is None:
            if hasattr(model, 'get_pbt_hparams'):
                self.tune_hparams = model.get_pbt_hparams()


        mp = _mp.get_context('forkserver')

        global_epoch = mp.Value('i', 0)

        # initialize population tasks
        population_tasks = mp.Queue(maxsize=self.population_size)
        for i in range(self.population_size):
            population_tasks.put({
                'id': i,
                self.pbt_monitor: 0,
                'checkpoint_path': None,
            })

        logger_info = dict(name=self.trainer.logger.name,
                           version=self.trainer.logger.version,
                           save_dir=self.trainer.logger.save_dir)

        workers = [Worker(
            pl_trainer=copy.deepcopy(self.trainer),
            model=copy.deepcopy(model),
            population_tasks=population_tasks,
            tune_hparams=self.tune_hparams,
            process_position=i,
            global_epoch=global_epoch,
            max_epoch=10,
            train_dataloader=copy.deepcopy(train_dataloader),
            val_dataloaders=copy.deepcopy(val_dataloaders),
            full_parallel=self.num_workers==self.population_size,
            logger_info=logger_info,
        ) for i in range(self.num_workers)]

        [w.start() for w in workers]
        [w.join() for w in workers]
        task = []

        while not population_tasks.empty():
            task.append(population_tasks.get())

    def test(self, model: Optional[LightningModule] = None, test_dataloaders: Optional[DataLoader] = None,):
        # Test on best task.
        model = model if model is not None else self.model
        self.trainer.test(self.model, test_dataloaders=test_dataloaders)
