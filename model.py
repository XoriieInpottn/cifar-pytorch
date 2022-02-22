#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2022-02-21
"""

import torch
from torch import nn

import backbone


class PSwish(nn.Module):

    def __init__(self, init=0.0):
        super(PSwish, self).__init__()
        self.weight = nn.Parameter(torch.empty(()).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.weight.sigmoid() + 0.5
        return x * (a * x).sigmoid()


def create_model(backbone_name: str, num_classes: int):
    fn = getattr(backbone, backbone_name)
    model = fn(num_classes=num_classes)

    def foo(m: nn.Module):
        for name, child in m.named_children():
            if isinstance(child, nn.ReLU):
                setattr(m, name, nn.SiLU())
            else:
                foo(child)

    # foo(model)
    return model
