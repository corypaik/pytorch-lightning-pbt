#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: Apr. 2020
# ---------------------

# General
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

extras = {
     'test': [
        'pytest',
        'pytest-cov',
        'pytest-flake8',
        'mypy',
    ],
}

extras['all'] = [iv for ov in extras.values() for iv in ov]

setup(
    name='lab',
    packages=['lab', ],
    install_requires=[
        'numpy',
        'pandas',
        'parse',
        'torch==1.5.0',
        'torchvision==0.6.0',
        'pytorch_lightning==0.7.5',
        'psutil',
        'tqdm',
    ],
    extras_require=extras,
    version='0.0.1',
    author='Cory Paik, Rocket Romero',
    author_email='corypaik@gmail.com, rocketromero4444@gmail.com',
    url='https://github.com/corypaik/pytorch-lightning-pbt/pytorch_lightning_pbt_examples',
    description='PyTorch Lightning PBT Examples.',
    long_description=open('README.md').read(),
)
