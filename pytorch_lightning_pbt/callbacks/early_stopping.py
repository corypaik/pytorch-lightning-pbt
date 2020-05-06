#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """

import torch.multiprocessing as mp
from pytorch_lightning.callbacks.base import Callback


class EarlyStopping(Callback):

    def __init__(self, global_epoch: mp.Value, max_global_epoch: int, verbose: bool = False):
        """

        Args:
            global_epoch:
            max_global_epoch:
            verbose:
        """
        super().__init__()
        # Set
        self.verbose = verbose
        self.stopped_epoch = 0
        self.wait = 0
        self.patience = 0
        self.global_epoch = global_epoch
        self.max_global_epoch = max_global_epoch

    # noinspection PyMethodMayBeStatic
    def check_metrics(self, logs):
        """ Dummy check_metrics. Always returns True as we only stop training at end of epoch.

        Args:
            logs:

        Returns:

        """
        return True

    def on_train_start(self, trainer, pl_module) -> None:
        """To be reused

        Args:
            trainer:
            pl_module:

        Returns:

        """
        # Allow instances to be re-used
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, pl_module) -> bool:
        """ Checks global_epoch to see if this pl_module should quit.

        Args:
            trainer:
            pl_module:

        Returns:
            True if the global_epoch > max_global_epoch, else False
        """
        stop_training = False
        if self.global_epoch.value > self.max_global_epoch:
            self.stopped_epoch = trainer.current_epoch
            stop_training = True
        return stop_training

    def on_train_end(self, trainer, pl_module):
        """ Early Stopping

        Args:
            trainer:
            pl_module:

        Returns:

        """
        if self.stopped_epoch > 0 and self.verbose > 0:
            log.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')
