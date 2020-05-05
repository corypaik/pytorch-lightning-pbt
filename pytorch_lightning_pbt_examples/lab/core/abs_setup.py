#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" Common Setup Utils """

import os
import random
from importlib import import_module

import lab
import numpy as np
import torch
from lab import _logger as log
from lab.core import arg_parser


def setup_run(*, args, mode=None):
    """ Sets up an experiment

        Arg parsing: This function creates and passes user defined arguments

        Creation of experiment run_dir, it can also take a specific run_code as a command line option
            to continue training or load an old expirment

    Args:
        args: should be passed as args=sys.argv
        mode: passed by file.

    Returns:
        alg_module: algorithm module containing the `Agent` class
        hparams: parser arguments as hyperparameters
        device: PyTorch Device

    """
    # Correct base run dir
    root_dir = os.path.split(os.path.dirname(lab.__file__))[0]
    os.chdir(root_dir)

    # Setup the argument parser
    alg_module, hparams = _setup_algorithm(args, mode=mode)

    # Setup Frameworks
    hparams, device = _setup_frameworks(hparams)

    return alg_module, hparams, device


# ------------------------------------------
# Backend
# ------------------------------------------
def _setup_algorithm(args, mode):
    # Setup the argument parser
    base_parser = arg_parser.common_arg_parser(mode=mode)
    b_args, _ = base_parser.parse_known_args(args[1:], arg_parser.Nestedspace())
    # set defaults from parsed args so they're readable by the agent classes.
    base_parser.set_defaults(**b_args.__unwrapped__)
    # load up the algorithm module
    alg_module = import_module('.'.join(['lab', 'ptl_agents', b_args.alg]))
    # add agent args and create full hparams
    parser = alg_module.Agent.add_module_args(base_parser)
    # Note that passed args will still overwrite defaults set by agent class since we re-parse here.
    hparams, unknown_args = parser.parse_known_args(args[1:], arg_parser.Nestedspace())
    # assert dataset arg present
    assert hparams.dataset != '', \
        f'Must provide env key or dataset key, but not both.'
    # allow mode override
    if mode is not None:
        hparams.mode = mode
    # unknown args are not allowed
    if unknown_args:
        _logger.critical('Unknown Args: %s', ''.join(x for x in unknown_args))
    return alg_module, hparams


def _setup_frameworks(hparams):
    """ Set up Framework Customizations and set seeds
        Adds debugging parameters to torch if --log=debug

    Args:
        hparams: hyperparameters (from arg parser)
    """
    if hparams.seed == -1:
        hparams.seed = random.randint(0, 10_000)

    # Seeding
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    # Torch setup
    torch.autograd.set_detect_anomaly(hparams.log.startswith('debug'))
    # Get torch device
    if torch.cuda.is_available() and hparams.gpu_idx >= 0:
        if hparams.gpu_idx != 0:
            raise NotImplementedError
        hparams.gpu_idx = hparams.gpu_idx if hparams.gpu_idx < torch.cuda.device_count() else 0
        device = torch.device('cuda', hparams.gpu_idx)
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(1 if hparams.log.startswith('debug') else 0)
    else:
        device = torch.device('cpu')
        log.warning('No GPU available, torch using cpu')

    return hparams, device
