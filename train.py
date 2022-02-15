#!/usr/bin/env python3


import argparse
import os
import shutil

import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchcommon.optim.lr_scheduler import CosineWarmUpAnnealingLR
from tqdm import tqdm

import dataset
from evaluate import ClassificationMeter
from models import ResNet18

cv.setNumThreads(0)


class Trainer(object):

    def __init__(self,
                 *,
                 data_path,
                 max_lr,
                 weight_decay,
                 batch_size,
                 num_epochs,
                 output_dir,
                 **kwargs):
        self.data_path = data_path
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._create_dataset()
        self._create_model()
        self._create_optimizer()

    def _create_dataset(self):
        self.train_loader = dataset.create_data_loader(
            os.path.join(self.data_path, 'train.ds'),
            self.batch_size,
            train=True,
        )
        self.test_loader = dataset.create_data_loader(
            os.path.join(self.data_path, 'test.ds'),
            self.batch_size,
            train=False
        )

    def _create_model(self):
        self.model = ResNet18()

        def foo(m: nn.Module):
            for name, child in m.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(m, name, nn.SiLU())
                else:
                    foo(child)

        foo(self.model)
        self.model = self.model.to(self.device)

    def _create_optimizer(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)
        self.optimizer = AdamW(
            params,
            lr=self.max_lr,
            weight_decay=self.weight_decay
        )
        num_loops = len(self.train_loader) * self.num_epochs
        self.scheduler = CosineWarmUpAnnealingLR(self.optimizer, num_loops)

    def train(self, image, label):
        image = image.to(self.device)
        label = label.to(self.device)

        output = self.model(image)
        output = F.softmax(output, -1)
        target = F.one_hot(label, output.shape[1]).float()
        loss = -((output * target).sum(-1) + 1e-10).log().mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        loss = loss.detach().cpu()
        return loss

    def predict(self, image):
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            label = output.argmax(-1)

            label = label.detach().cpu()
            return label

    def run(self):
        loss_g = None
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loop = tqdm(self.train_loader, leave=False, ncols=0)
            for doc in train_loop:
                image = doc['image']
                label = doc['label']
                loss = self.train(image, label)

                loss_g = 0.9 * loss_g + 0.1 * float(loss) if loss_g is not None else float(loss)
                lr = self.scheduler.get_last_lr()[0]
                info = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                train_loop.set_description(info, False)

            # if (epoch + 1) % 2 != 0 and (epoch + 1) != self.num_epochs:
            #     continue

            meter = ClassificationMeter(10)
            self.model.eval()
            test_loop = tqdm(self.test_loader, leave=False, ncols=0)
            for doc in test_loop:
                image = doc['image']
                output = self.predict(image)
                target = doc['label']
                meter.add(output.numpy(), target.numpy())
            tqdm.write(
                f'[{epoch + 1}/{self.num_epochs}] '
                f'L={loss_g:.06f} '
                f'ACC={meter.accuracy_score():.02%}'
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data-path', required=True)

    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)

    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--note')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.output_dir is not None:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    kwargs = {
        name: getattr(args, name)
        for name in dir(args)
        if not name.startswith('_')
    }
    Trainer(**kwargs).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
