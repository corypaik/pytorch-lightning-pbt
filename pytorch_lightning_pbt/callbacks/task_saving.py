#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""
"""

import os
import re

import parse
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn

from pytorch_lightning_pbt import _logger as log
from pytorch_lightning_pbt.callbacks import TaskIOMixin

class TaskSaving(ModelCheckpoint, TaskIOMixin):
    """ Task saving for Population Based Training. """
    def __init__(self, full_parallel: bool, population_tasks: mp.Queue, filepath: str, **kwargs):
        """ Note certain Args have slightly different uses here.

        Args:
            population_tasks:
            filepath:
            **kwargs:

        Keyword Args:
            period: Note this is normally the "Interval (number of epochs) between checkpoints."
                In Population Based Training we also use this to determine how many epochs to take for
                a member of the population to be determined as 'Ready'.

                For example, passing `period = 3` means that each member of the population will run 3 local
                epochs prior to running an exploration and exploitation step.
        """
        super(TaskSaving, self).__init__(filepath, **kwargs)

        self.population_tasks = population_tasks
        self.full_parallel = full_parallel

    def on_validation_end(self, trainer, pl_module):
        """

        Args:
            trainer:
            pl_module:

        Returns:

        """
        # only run on main process
        if trainer.proc_rank != 0:
            return
        # Set metrics and epoch
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        current = metrics.get(self.monitor)
        # check current
        if current is None:
            rank_zero_warn(
                f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning
            )
        # new epoch
        self.epoch_last_check = epoch
        # here we simply check by task and replace the one with the same task by this model.
        remove_files = list(filter(lambda x: x['id'] == pl_module._pbt_task_id, self._get_sorted_tasks()))
        if len(remove_files) > 1:
            log.critical(f'Found multiple files with the same task id: {remove_files}')
            [self._del_model(os.path.join(file['checkpoint_path'])) for file in remove_files]
        elif len(remove_files) == 1:
            self._del_model(os.path.join(remove_files[0]['checkpoint_path']))

        metrics['task'] = pl_module._pbt_task_id
        filepath = self.format_checkpoint_name(epoch, metrics)   # set filepath at checkpoint

        log.debug(f'trainer {trainer.process_position} saving model.')
        self._save_model(filepath)
        current = metrics.get(self.monitor)

        # Place this as a finished task on the queue if the size of our population is > num_wokers.
        if not self.full_parallel:
            td = {'id': pl_module._pbt_task_id, self.monitor: current, 'checkpoint_path': filepath}
            self.population_tasks.put(td)
            pl_module._pbt_task_id = None
