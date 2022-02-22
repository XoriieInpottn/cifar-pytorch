#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2022-02-21
"""

import abc
from collections import OrderedDict
from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchcommon.optim.lr_scheduler import CosineWarmUpAnnealingLR
from tqdm import tqdm

import dataset
from torchstocks.utils.metrics import ClassificationMeter


class AbstractTester(abc.ABC):

    def inference(self, x: torch.Tensor):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class AbstractTrainer(abc.ABC):

    def train(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class Tester(AbstractTester):

    def __init__(self,
                 model: nn.Module,
                 data_path: Union[str, List[str]],
                 batch_size: int,
                 device):
        super(Tester, self).__init__()
        self.model = model
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device

        self.data_loader = DataLoader(
            dataset.Cifar10Dataset(self.data_path, False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )

    def inference(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            y = output.argmax(-1)

            y = y.detach().cpu()
            return y

    def run(self):
        meter = ClassificationMeter()
        self.model.eval()
        test_loop = tqdm(self.data_loader, leave=False, ncols=96)
        for doc in test_loop:
            x = doc['image']
            y_true = doc['label']
            # x, y_true = doc
            y_pred = self.inference(x)
            meter.add(y_pred.numpy(), y_true.numpy())
        result = OrderedDict(acc=meter.accuracy_score())
        return result


class Trainer(AbstractTrainer):

    def __init__(self,
                 model: nn.Module,
                 train_data_path: Union[str, List[str]],
                 test_data_path: Union[str, List[str]],
                 optimizer: str,
                 max_lr: float,
                 min_lr: float,
                 weight_decay: float,
                 momentum: float,
                 batch_size: int,
                 num_epochs: int,
                 output_dir: str,
                 device):
        super(Trainer, self).__init__()
        self.model = model
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.optimizer_name = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = device

        self._create_dataset()
        self._create_optimizer()
        self._create_tester()

    def _create_dataset(self):
        self.train_loader = DataLoader(
            dataset.Cifar10Dataset(self.train_data_path, True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True
        )

    def _create_optimizer(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)
        from torch import optim
        opt_class = getattr(optim, self.optimizer_name, None)
        if opt_class is None:
            from torchstocks import optim
            opt_class = getattr(optim, self.optimizer_name, None)
        assert opt_class is not None
        self.optimizer = opt_class(
            params,
            lr=self.max_lr,
            weight_decay=self.weight_decay,
            betas=(self.momentum, 0.999)
        )
        print(type(self.optimizer))
        num_loops = len(self.train_loader) * self.num_epochs
        self.scheduler = CosineWarmUpAnnealingLR(
            self.optimizer,
            num_loops=num_loops,
            min_factor=self.min_lr / self.max_lr
        )

    def _create_tester(self):
        self.tester = Tester(
            self.model,
            data_path=self.test_data_path,
            batch_size=self.batch_size,
            device=self.device
        ) if self.test_data_path else None

    def train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.model(x)
        output = F.softmax(output, -1)
        target = F.one_hot(y, output.shape[-1]).float()
        loss = -((output * target).sum(-1) + 1e-10).log().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss = loss.detach().cpu()
        return loss

    def run(self):
        loss_g = None
        for epoch in range(self.num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                x = doc['image']
                y_true = doc['label']
                # x, y_true = doc
                loss = self.train(x, y_true)

                loss_g = 0.9 * loss_g + 0.1 * float(loss) if loss_g is not None else float(loss)
                lr = self.scheduler.get_last_lr()[0]
                info = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)

            info = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f}'
            if self.tester:
                info += ''.join([f' {k}={v:.04f}' for k, v in self.tester.run().items()])
            tqdm.write(info)
