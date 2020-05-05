<div align="center">

# Pytorch Lightning Population Based Training  
[![CircleCI](https://circleci.com/gh/corypaik/pytorch-lightning-pbt.svg?style=shield&circle-token=9400bcc91d4ae20ea1cf7526c1d84b6cdc759210)](https://circleci.com/gh/corypaik/pytorch-lightning-pbt) 
[![codecov](https://codecov.io/gh/corypaik/pytorch-lightning-pbt/branch/master/graph/badge.svg?token=2yDGmpQZ7h)](https://codecov.io/gh/corypaik/pytorch-lightning-pbt)
[![CodeFactor](https://www.codefactor.io/repository/github/corypaik/pytorch-lightning-pbt/badge)](https://www.codefactor.io/repository/github/corypaik/pytorch-lightning-pbt)

</div>

---
This repository is an extension meant to simplify Population Based Training using [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning).

Currently we are implementing this as a standalone project, but most all of the code is based directly on PyTorch Lightning
and we hope to integrate this into that package as we develop better testing and integration. 

This project is still very much in a development phase, but if you'd like to try it out we've included 
some setup instructions [below](#setup), and some starter examples [here](pytorch_lightning_pbt_examples). 

Currently this implementation supports the truncation method and performs perturbations on user-defined hyperparameters.


# Setup
```bash
# core package
pip install -e .
# examples
pip install -e .pytorch_lightning_pbt_examples/
```


# Credits
[PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning)  
[Population Based Training](https://deepmind.com/blog/article/population-based-training-neural-networks)
