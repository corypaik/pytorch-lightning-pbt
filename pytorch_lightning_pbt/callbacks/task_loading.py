#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """
import inspect
import os
import re
from argparse import Namespace
import numpy as np
import parse
import torch
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pytorch_lightning_pbt import _logger as log
from pytorch_lightning_pbt.callbacks import TaskIOMixin


class TaskLoading(ModelCheckpoint, TaskIOMixin):

    def __init__(self, pbt_period, population_tasks: mp.Queue, global_epoch: mp.Value, tune_hparams,
                 filepath: str, **kwargs):
        """

        Args:
            pbt_period:
            population_tasks:
            global_epoch:
            tune_hparams:
            filepath:
            **kwargs:
        """
        super(TaskLoading, self).__init__(filepath, **kwargs)
        # Initilize
        self.population_tasks = population_tasks
        self.global_epoch = global_epoch
        self.tune_hparams = tune_hparams
        self.cutoff_frac = 0.2
        self.pbt_period = pbt_period

    def on_validation_end(self, trainer, pl_module):
        """ Checks on validation end

        Args:
            trainer:
            pl_module:

        Returns:

        """
        return

    def on_train_start(self, trainer, pl_module):
        """ Checks that the pl_module does not have a _pbt_task_id attribute.
        Args:
            trainer:
            pl_module:

        Returns:

        """

        if getattr(pl_module, '_pbt_task_id', -1) != -1:
            raise MisconfigurationException(
                f'Cannot perform Population Based Training. \n'
                f'Your module, {pl_module.__class__.__name__}, has initiated attribute `_pbt_task_id` '
                f'to a value other than -1. \n'
                f'Note that initiating the value to `-1` or reading from the value is okay, '
                f'however writing to this value during training will cause serious problems. ')

        pl_module._pbt_task_id = None

        # TODO: verify tune_hparams.
        # Set checkpoint
        checkpoint = trainer.dump_checkpoint()
        self._pertrub_checkpoint(trainer, checkpoint)

    def _pertrub_checkpoint(self, trainer, checkpoint, perturb_factors=(1.2, 0.8)):
        """Pertrub checkpoint with initilized factors

        Args:
            trainer:
            checkpoint:
            perturb_factors:

        Returns:

        """
        # raw hparams:
        if self.tune_hparams.get('hparams', None) is not None:
            if checkpoint['hparams'] is None:
                raise MisconfigurationException(
                    f"PBT was configured to tune hparams {self.tune_hparams['hparams']}, "
                    f"but no hparams was found in the checkpoint.\n")
            assert checkpoint['hparams'] is not None
            # Tune hparams
            for tp_key in self.tune_hparams['hparams']:
                tp_type = type(checkpoint['hparams'][tp_key])
                checkpoint['hparams'][tp_key] *= np.random.choice(perturb_factors)
                checkpoint['hparams'][tp_key] = tp_type(checkpoint['hparams'][tp_key])

        # optimizers check
        if self.tune_hparams.get('optimizer_states', None) is not None:
            if len(self.tune_hparams['optimizer_states']) != len(checkpoint['optimizer_states']):
                raise MisconfigurationException(
                    f"PBT was configured to tune {len(self.tune_hparams['optimizer_states'])} optimizers,"
                    f"but your model implements {len(checkpoint['optimizer_states'])}. ")

            # for each optimizer
            for optimizer_idx, (optimizer_tps, optimizer_state) in \
                    enumerate(zip(self.tune_hparams['optimizer_states'], checkpoint['optimizer_states'])):
                for tps in optimizer_tps:
                    # for each param group | pertrub the same on each optimizer
                    perturb = np.random.choice(perturb_factors)
                    for param_group in optimizer_state['param_groups']:
                        if tps not in param_group.keys():
                            raise MisconfigurationException(
                                f"PBT was configured to tune parameter {tps} in optimizer {optimizer_idx} "
                                f"of type {trainer.optimizers[optimizer_idx].__class__}, but "
                                f"this optimizer does not have that parameter. ")
                        param_group[tps] *= perturb

        return checkpoint

    def _exploit_and_explore(
            self,
            trainer,
            pl_module,

            top_task,
            perturb_factors=(1.2, 0.8)
    ):

        """Copy parameters from the better model and the hyperparameters
                       and running averages from the corresponding optimizer."""

        # load the new (top) checkpoint
        checkpoint = torch.load(top_task['checkpoint_path'])
        # apply pertrub checkpoint
        checkpoint = self._pertrub_checkpoint(trainer, checkpoint, perturb_factors)
        self._load_task_checkpoint(pl_module, checkpoint=checkpoint)

        with self.global_epoch.get_lock():
            self.global_epoch.value += 1

    def _get_training_task(self, trainer, pl_module):
        """Waits for a task

        Args:
            trainer:
            pl_module:

        Returns:

        """
        task = self.population_tasks.get()
        return task

    def _maybe_exploit_and_explore(self, trainer, pl_module):
        """ Check the status of the task

        Args:
            trainer:
            pl_module:

        Returns:

        """
        # either it needs training or it needs exploit/explore.
        # check if the task is task score is in top 80%. if so, just load it and train.
        # tasks are sorted best to worst.
        sorted_tasks = self._get_sorted_tasks()
        task_rankings = [t['id'] for t in sorted_tasks]
        # sort
        s, e = int(np.ceil(self.cutoff_frac * len(sorted_tasks))), \
               int(np.ceil((1 - self.cutoff_frac) * len(sorted_tasks)))

        try:
            task_ranking = task_rankings[:e].index(pl_module._pbt_task_id)
        except ValueError:
            # bottom 20% -> sample a task from the top 20%
            self._exploit_and_explore(trainer, pl_module,
                                      top_task=np.random.choice(sorted_tasks[:s]),)
        # else:
        #     self._load_task_checkpoint(pl_module, checkpoint_path=task['checkpoint_path'])

    def on_epoch_start(self, trainer, pl_module):
        """

        Args:
            trainer:
            pl_module:

        Returns:

        """
        # Use val_interval.
        log.debug(f'In Population on_epoch_start, epoch={trainer.current_epoch}')
        if pl_module._pbt_task_id is None:
            task = self._get_training_task(trainer, pl_module)
            # set this module as fulfilling that task.
            pl_module._pbt_task_id = task['id']
            # no loading.
            if task['checkpoint_path'] is None:
                return
            self._load_task_checkpoint(pl_module, checkpoint_path=task['checkpoint_path'])

        # now that we're loaded up, check exploit / explore
        if trainer.current_epoch % self.pbt_period == 0:
            self._maybe_exploit_and_explore(trainer, pl_module)

    @staticmethod
    def _load_task_checkpoint(pl_module, checkpoint=None, checkpoint_path=None, map_location=None):
        """load checkpoint

        Args:
            pl_module:
            checkpoint:
            checkpoint_path:
            map_location:

        Returns:

        """

        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # try to get hparams
        if pl_module.hparams is not None:
            pl_module.hparams = checkpoint.get('hparams', None)

        pl_module.load_state_dict(checkpoint['state_dict'])
        # give model a chance to load something
        pl_module.on_load_checkpoint(checkpoint)
