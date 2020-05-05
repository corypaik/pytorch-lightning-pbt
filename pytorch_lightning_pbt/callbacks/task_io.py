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
import re
import parse

from pytorch_lightning_pbt import _logger as log


class TaskIOMixin:
    prefix: str
    filename: str
    monitor: str
    mode: str
    dirpath: str

    def _get_parse_fmt_str(self):
        """Generate a filename according to the defined template.

        """
        # check if user passed in keys to the string
        groups = re.findall(r'({.*?)[:\}]', self.filename)
        filename = self.filename
        for tmp in groups:
            name = tmp[1:]
            filename = filename.replace(tmp, name + '={' + name)
        return self.prefix + filename + '.ckpt'

    def _get_sorted_tasks(self):
        """ Returns a list of tuples: (checkpoint path, task, score)

        """
        files = [file for file in os.listdir(self.dirpath) if file.endswith('.ckpt')]
        tasks = []
        log.debug(self.prefix + self.filename + '.ckpt')
        # get results for every file
        for file in files:
            result = parse.parse(self._get_parse_fmt_str(), file)
            if result:
                tasks.append({
                    'checkpoint_path': os.path.join(self.dirpath, file),
                    'id': result['task'],
                    self.monitor: result[self.monitor]})
            else:
                log.warning(f'Unable to parse checkpoint filename: {file}')
        return sorted(tasks, key=lambda x: x[self.monitor], reverse=self.mode == 'max')
