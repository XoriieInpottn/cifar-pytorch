#!/usr/bin/env python3

"""
@author: xi
@since: 2022-10-08
"""

from torch import nn
from torchcommon import BaseConfig
from torchcommon.models import cifar


class ModelConfig(BaseConfig):

    def __init__(self):
        self.backbone = None
        self.non_lin = None
        self.num_classes = None


def Model(config: ModelConfig):
    Backbone = getattr(cifar, config.backbone)
    NonLin = getattr(nn, config.non_lin) if config.non_lin is not None else None
    backbone = Backbone(num_classes=config.num_classes)
    _replace_non_lin(backbone, NonLin)
    return backbone


def _replace_non_lin(module: nn.Module, NonLin: nn.Module):
    for name, child in module.named_children():
        if NonLin is not None and isinstance(child, nn.ReLU):
            setattr(module, name, NonLin())
        _replace_non_lin(child, NonLin)
