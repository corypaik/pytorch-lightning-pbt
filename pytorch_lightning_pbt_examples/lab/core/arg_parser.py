#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------


import argparse

from collections import Mapping


def common_arg_parser(mode=None):
    """ Creates an arg parser of top level args for run modes, algorithm, environment, and run codes
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_core_params(parser)

    # if mode provided, will only add args for that mode
    if mode in ('train', 'tune', None):
        add_train_params(parser)
    if mode in ('test', 'play', None):
        add_test_params(parser)
    if mode in ('tune', None):
        add_tune_params(parser)

    add_project_params(parser)
    return parser


# ---------------------
# PROJECT PARAMS
# ---------------------
def add_project_params(parser):
    # Training
    parser.add_argument('--total_timesteps', type=int, default=1.e+7)
    parser.add_argument('--nupdates', type=int, default=-1)

    parser.add_argument('--nsteps', default=128, type=int)
    parser.add_argument('--nminibatches', default=4, type=int)
    parser.add_argument('--nepochs_per_update', default=4, type=int)

    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--use_pbt', default=0, type=int)

    parser.add_argument('--population_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=512, )
    parser.add_argument('--val_split', type=float, default=0.2, )
    parser.add_argument('--shuffle', type=int, default=1, )
    parser.add_argument('--dl_num_workers', type=int, default=1, )

# ---------------------
# CORE PARAMS
# ---------------------
def add_core_params(parser):
    """ Adds core configuration parameters.
        These are independent of project add contribute to core features of the package.

    Args:
        parser: argparse.ArgumentParser object

    """
    parser.add_argument('--alg', help='algorithm name', type=str, default='ppo')
    parser.add_argument('--mode', help='run mode', type=str, default='train',
                        choices=['train', 'test', 'tune'])
    parser.add_argument('--version', help='Choose a version #', type=int)
    parser.add_argument('--seed', help='Set seeds, -1 for random (seed will be logged)', type=int, default=-1)
    parser.add_argument('--gpu_idx', help='GPU index, -1 for cpu', type=int, default=0)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--log_args', help='Log User Args', type=int, default=1)

    parser.add_argument('--clear_runs', help='Clear runs in dir', type=str, default='local',
                        choices=['local', 'alg', 'global'])
    parser.add_argument('--clear_cond', help='Clear with condition', type=str, default='none',
                        choices=['none', 'ckpt', 'results', 'all'])

    parser.add_argument('--log', help='Enter logger level', default='info',
                        choices=['debug_shapes', 'debug', 'info', 'warn', 'error', 'none'], type=str)
    parser.add_argument('--ckpt', default=1, type=int)
    parser.add_argument('--progressbar', help='Progressbar', type=int, default=1)
    parser.add_argument('--monitor', default='val_loss', type=str)
    parser.add_argument('--pytest', default=0, type=int)
    parser.add_argument('--filter_warnings', default=1, type=int)


def add_train_params(parser):
    parser.add_argument('--nepochs', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--val_percent_check', default=1., type=float)
    parser.add_argument('--val_interval', default=1., type=float)
    parser.add_argument('--reload_dataloaders_every_epoch', default=1, type=int,
                        help='Reload dataloaders. This should be true unless using Dataset (eg. mnist)')


# ---------------------
# BACKEND
# ---------------------
class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    @property
    def __unwrapped__(self):
        udict = {}

        def _nd_to_dict(_val, _scope):
            # dictionaries
            if isinstance(_val, Mapping):
                prev_scope = f'{_scope}.' if _scope != '' else ''
                for k, v in _val.items():
                    new_scope = f'{prev_scope}{k}'
                    _nd_to_dict(v, _scope=new_scope)
            # Nestedspace and Namespace
            elif isinstance(_val, Nestedspace):
                _nd_to_dict(_val.__dict__, _scope=_scope)
            # Unwrapped
            else:
                udict[_scope] = _val

        _nd_to_dict(_val=self.__dict__, _scope='')

        return udict

