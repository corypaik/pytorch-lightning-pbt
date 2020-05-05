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
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from typing import Optional, Union, Dict
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning_pbt import _logger as log


class TensorBoardLogger(pl.loggers.TensorBoardLogger):
    """ Slight alteration pytorch_lightning version """
    def __init__(self,
                 save_dir: str,
                 task: Optional[int] = None,
                 task_prefix: Optional[str] = 'worker',
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 **kwargs):
        super().__init__(save_dir, name, version, **kwargs)
        self._task = task
        self._task_prefix = task_prefix

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """ Nest certain objects. """

        for k, v in metrics.items():

            # deal with keys
            out_key = k
            if k.startswith('grad_'):
                out_key = f'grads/{k}'
            elif k.startswith('gpu_'):
                out_key = f'gpus/{k}'
            # -------------
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(out_key, v, step)

    @property
    def log_dir(self) -> str:
        """
        The directory for this run's tensorboard checkpoint. By default, it is named
        ``'version_${self.version}'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        task = self.task if isinstance(self.task, str) else f"{self._task_prefix}_{self.task}"
        log_dir = os.path.join(self.root_dir, version, task)
        return log_dir

    @property
    def task(self) -> int:
        if self._task is None:
            self._task = self._get_next_task()
        return self._task

    def _get_next_task(self):
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"

        version_dir = os.path.join(self.save_dir, self.name, version)

        if not os.path.isdir(version_dir):
            log.warning('Missing version logger folder: %s', version_dir)
            return 0

        existing_tasks = []
        for d in os.listdir(version_dir):
            if os.path.isdir(os.path.join(version_dir, d)) and d.startswith(f"{self._task_prefix}_"):
                existing_tasks.append(int(d.split("_")[1]))

        if len(existing_tasks) == 0:
            return 0

        return max(existing_tasks) + 1




