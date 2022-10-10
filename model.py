#!/usr/bin/env python3

from torch import nn
from torchcommon import BaseConfig
from torchcommon import nn as nn_
from torchcommon.models import cifar


class ModelConfig(BaseConfig):

    def __init__(self):
        self.backbone = None
        self.non_lin = None
        self.norm = None
        self.num_classes = None


def Model(config: ModelConfig):
    Backbone = getattr(cifar, config.backbone)

    NonLin = None
    if config.non_lin is not None:
        for n in [nn, nn_]:
            NonLin = getattr(n, config.non_lin, None)
            if NonLin is not None:
                break

    Norm = None
    if config.norm is not None:
        for n in [nn, nn_]:
            Norm = getattr(n, config.norm, None)
            if Norm is not None:
                break

    backbone = Backbone(num_classes=config.num_classes)
    _replace_modules(backbone, NonLin, Norm)
    return backbone


def _replace_modules(module: nn.Module, NonLin, Norm):
    for name, child in module.named_children():
        if NonLin is not None and isinstance(child, nn.ReLU):
            setattr(module, name, NonLin())
        if Norm is not None and isinstance(child, nn.BatchNorm2d):
            if issubclass(Norm, nn.GroupNorm):
                norm = Norm(48, child.num_features)
            else:
                norm = Norm(child.num_features)
            setattr(module, name, norm)
        _replace_modules(child, NonLin, Norm)
