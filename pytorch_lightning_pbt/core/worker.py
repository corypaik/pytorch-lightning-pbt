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
import pytorch_lightning as pl
import torch.multiprocessing as _mp
from pytorch_lightning_pbt.callbacks import EarlyStopping
from pytorch_lightning_pbt.callbacks import TaskLoading
from pytorch_lightning_pbt.callbacks import TaskSaving
from pytorch_lightning_pbt import loggers
from typing import Dict, Optional
mp = _mp.get_context('spawn')

import pytorch_lightning_pbt as pbt



class Worker(mp.Process):
    """ Worker Process. """
    def __init__(self,
                 pl_trainer: pl.Trainer,
                 model: pl.LightningModule,
                 population_tasks: mp.Queue,
                 tune_hparams: Dict,
                 process_position: int,
                 global_epoch: mp.Value,
                 max_epoch: int,
                 full_parallel: bool,
                 pbt_period: int = 4,
                 pbt_monitor: str = 'val_loss',
                 logger_info = None,
                 dataloaders: Optional[Dict]= None):
        """

        Args:
            pl_trainer:
            model:
            population_tasks:
            tune_hparams:
            process_position:
            global_epoch:
            max_epoch:
            full_parallel:
            pbt_period:
            **dataloaders:
        """
        super().__init__()
        # Set monitor and monitor_precision
        monitor_precision = 32
        # Set checkpoint dirpath
        #checkpoint_dirpath = pl_trainer.checkpoint_callback.dirpath
        #period = pl_trainer.checkpoint_callback.period
        # Formatting checkpoints
        checkpoint_format = '{task:03d}-{' + f'{pbt_monitor}:.{monitor_precision}f' + '}'
        checkpoint_filepath = os.path.join(pl_trainer.logger.log_dir, checkpoint_format)

        # For TaskSaving
        print(logger_info)

        checkpoint_dirpath = pl_trainer.logger.log_dir

        pl_trainer.checkpoint_callback = TaskSaving(
            filepath=checkpoint_filepath,
            monitor=pbt_monitor,
            population_tasks=population_tasks,
            period=1,
            full_parallel=full_parallel,)

        # For EarlyStopping
        pl_trainer.early_stop_callback = EarlyStopping(
            global_epoch=global_epoch,
            max_global_epoch=max_epoch)

        # For TaskLoading
        pl_trainer.callbacks = [TaskLoading(
            population_tasks=population_tasks,
            global_epoch=global_epoch,
            filepath=checkpoint_filepath,
            monitor=pbt_monitor,
            tune_hparams=tune_hparams,
            pbt_period=pbt_period)]

        # Alter logger to spec.
        #if isinstance(pl_trainer.logger, pl.loggers.TensorBoardLogger):
        pl_trainer.logger = loggers.TensorBoardLogger(
            save_dir=logger_info['save_dir'],
            name=logger_info['name'],
            version=logger_info['version'],
            task=process_position,)


        # Set process_position
        pl_trainer.process_position = process_position
        # pl_trainer.logger._version = f'worker_{process_position}'
        # Define and set = to
        self.trainer = pl_trainer
        self.model = model
        self.global_epoch = global_epoch
        self.population_tasks = population_tasks
        self.max_epoch = max_epoch
        self.dataloaders = dataloaders or {}
        print(dataloaders)

    def run(self):
        """ run

        Returns:

        """
        self.trainer.fit(self.model, **self.dataloaders)
