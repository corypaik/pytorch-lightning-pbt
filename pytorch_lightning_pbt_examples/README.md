<div align="center">

# PyTorch Lightning PBT Example Implementations.

</div>

---
This package contains some example implementations of Population Based Training.

- [x] [Simple Multi Layer Perceptron (MNIST)](lab/ptl_agents/mlp_ds.py)
- [x] [Simple CNN (MNIST & CIFAR10)](lab/ptl_agents/resnet_ds.py)
- [x] [Auto-Encoding Variational Bayes (MNIST)](lab/ptl_agents/vae_ds.py)
- [x] [Deep Residual Learning for Image Recognition (CIFAR10)](lab/ptl_agents/resnet_ds.py)


### Setup (from this directory)
```bash
# core package (from base directory)
pip install -e ../.
# examples
pip install -e .
```

### Running the examples.
All of the examples run with the following format. 
```bash
python -m lab.train --alg=[mlp_ds, cnn_ds, resnet_ds] --dataset=[mnist, cifar10, cifar100] --use_pbt=[1, 0]
```

Note that not every model is fully compatible with each dataset, and the preconfigured defaults for each model work best.  

These defaults can be run with:
```
python -m lab.train --alg=[mlp_ds, cnn_ds, resnet_ds] --use_pbt=[1, 0]
```

where `use_pbt=1` uses population based training, and `use_pbt=0` does not use population based training. 


### Credits
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
