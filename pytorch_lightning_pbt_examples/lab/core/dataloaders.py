#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
"""  """

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import MNIST


def get_dl(hparams, ds_types='all'):
    """

    Args:
        hparams:
        ds_types: one of [train_val, all, test]

    Returns:

    """
    dataloaders = None

    # make a tuple.
    if ds_types == 'all':
        ds_types = ('train', 'val', 'test')
    elif ds_types == 'train_val':
        ds_types = ('train', 'val')
    elif ds_types == 'test':
        ds_types = ('test', )
    else:
        raise NotImplementedError

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if hparams.dataset == 'mnist':

        dataset = MNIST('~/datasets/mnist', train=True, download=True, transform=transforms.ToTensor())
        val_size = int(hparams.val_split * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
        if 'test' in ds_types:
            test_dataset = MNIST('~/datasets/mnist', train=False, download=True,
                                 transform=transforms.ToTensor())
    elif hparams.dataset == 'cifar10':
        dataset = CIFAR10('~/datasets/cifar10', train=True, download=True, transform=transforms.ToTensor())
        val_size = int(hparams.val_split * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
        if 'test' in ds_types:
            test_dataset = CIFAR10('~/datasets/cifar10', train=False, download=True,
                                 transform=transforms.ToTensor())

    elif hparams.dataset == 'cifar100':
        dataset = CIFAR100('~/datasets/cifar100', train=True, download=True, transform=transforms.ToTensor())
        val_size = int(hparams.val_split * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
        if 'test' in ds_types:
            test_dataset = CIFAR100('~/datasets/cifar100', train=False, download=True,
                                   transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    dataloaders = dict(
        train_dataloader=None if 'train' not in ds_types else
        DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=hparams.shuffle, num_workers=hparams.dl_num_workers),
        val_dataloaders=None if 'val' not in ds_types else
        DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.dl_num_workers),
        test_dataloaders=None if 'test' not in ds_types else
        DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.dl_num_workers),)

    return {k:v for k,v in dataloaders.items() if v is not None}
