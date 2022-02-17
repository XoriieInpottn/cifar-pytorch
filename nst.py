#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2022-02-16
"""
import argparse
import os
from typing import List

import cv2 as cv
import numpy as np
import torch
import torchcommon.optim.lr_scheduler
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm


class VGGAdapter(nn.Module):

    def __init__(self, model: nn.Module):
        super(VGGAdapter, self).__init__()
        self._model = model

        self._y_list = []
        for layer in self._model.features:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._forward_hook)

        mode = self.training
        self.train(False)
        dummy = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        self.sizes = [y.shape[1] for y in self(dummy)]
        self.train(mode)

    def _forward_hook(self, _module, _x, y):
        self._y_list.append(y)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._model(x)
        y_list = self._y_list
        self._y_list = []
        return y_list


class ImageNet(object):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
    STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

    @staticmethod
    def encode_image(image: np.ndarray) -> np.ndarray:
        image = np.array(image, dtype=np.float32)
        image -= ImageNet.MEAN
        image /= ImageNet.STD
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def decode_image(tensor: np.ndarray) -> np.ndarray:
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor *= ImageNet.STD
        tensor += ImageNet.MEAN
        np.clip(tensor, 0, 255, out=tensor)
        return np.array(tensor, dtype=np.uint8)


def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    g.div_(h * w)
    return g


def load_image(path):
    image = cv.imread(path, cv.IMREAD_COLOR)
    image = cv.resize(image, (512, 512))
    cv.cvtColor(image, cv.COLOR_BGR2RGB, image)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from torchvision.models import vgg19
    backbone = VGGAdapter(vgg19(pretrained=True)).to(device)
    backbone.train(False)
    for p in backbone.parameters():
        p.requires_grad = False

    content_image = load_image('content.jpg')
    style_image = load_image('style.jpg')

    content_image = torch.tensor(ImageNet.encode_image(content_image)[None, ...]).to(device)
    style_image = torch.tensor(ImageNet.encode_image(style_image)[None, ...]).to(device)
    opt_image = Variable(content_image.clone(), requires_grad=True)

    content_layers = [9]
    style_layers = [0, 2, 4, 8, 12]

    content_targets = backbone(content_image)
    content_targets = [content_targets[i].detach() for i in content_layers]
    style_target = backbone(style_image)
    style_target = [gram_matrix(style_target[i]).detach() for i in style_layers]

    content_weights = [1e0]
    style_weights = [1e3 / backbone.sizes[i] ** 2 for i in style_layers]

    optimizer = SGD([opt_image], lr=1e-2, momentum=0.9)
    loop = tqdm(range(20000))
    for i in loop:
        outs = backbone(opt_image)
        content_outs = [outs[i] for i in content_layers]
        style_outs = [outs[i] for i in style_layers]
        content_loss = torch.stack([
            F.mse_loss(o, t) * w
            for o, t, w in zip(content_outs, content_targets, content_weights)
        ]).sum()
        style_loss = torch.stack([
            F.mse_loss(gram_matrix(o), t) * w
            for o, t, w in zip(style_outs, style_target, style_weights)
        ]).sum()
        loss = content_loss + style_loss * 5e5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = float(loss)
        loop.set_description(f'L={loss:.06f}', False)
        if (i + 1) % 50 == 0:
            loop.write(f'L={loss:.06f}')

            image = ImageNet.decode_image(opt_image.detach().to('cpu').numpy()[0])
            cv.cvtColor(image, cv.COLOR_RGB2BGR, image)
            cv.imwrite('output.jpg', image)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
