#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------
#  PyTorch Lightning PBT
#  Authors: Cory Paik
#           Rocket Romero
#  Updated: May. 2020
# ---------------------
""" ResNet Network in PyTorch.

    The network architecture is based on the original paper for CIFAR-10 [1].
    This implementation has been copied almost directly form the torchvision implementation [2] to process images of this size.
    The torchvision implementation only supports images "where H and W are expected to be at least 224."
Source:
    [1] He, K., Zhang, X., Ren, S. & Sun, J. Deep Residual Learning for Image Recognition.
        2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016) doi:10.1109/cvpr.2016.90.
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import conv1x1


class ResNetDownsampleA(nn.Module):
    def __init__(self, planes):
        super(ResNetDownsampleA, self).__init__()
        self._planes = planes

    def forward(self, x):
        return F.pad(input=x[:, :, ::2, ::2], pad=(0, 0, 0, 0, self._planes // 4, self._planes // 4),
                     mode='constant', value=0)


class ResNet(nn.Module):
    """ Modified version of torchvision.models.resnet """
    size_map = {
        20: [3, 3, 3],
        32: [5, 5, 5],
        44: [7, 7, 7],
        56: [9, 9, 9],
        110: [18, 18, 18],
        1202: [100, 200, 200]}

    def __init__(self, size, block=None, num_classes=10, downsample_type='A'):
        super(ResNet, self).__init__()

        self._downsample_type = downsample_type

        block = block or BasicBlock
        blocks = self.size_map[size]

        self.inplanes = 16
        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, 16, blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

        # Initialize Weights & Biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # downsample only on first block if stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Paper's CIFAR-10 method
            if self._downsample_type == 'A':
                downsample = ResNetDownsampleA(planes=planes)
            # torchvision method
            elif self._downsample_type == 'B':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    self._norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride,
                            downsample=downsample, norm_layer=self._norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
