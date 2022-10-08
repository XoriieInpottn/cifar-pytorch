#!/usr/bin/env python3
import math

import torch
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchcommon import BaseConfig
from torchcommon.optim import LRScheduler, CosineWarmupDecay
from torchcommon.utils.metrics import ClassificationMeter
from tqdm import tqdm


class TrainerConfig(BaseConfig):

    def __init__(self):
        super(TrainerConfig, self).__init__()

        self.model = None
        self.criterion = None

        self.train_dataset = None
        self.test_dataset = None

        self.optimizer = 'AdamW'
        self.batch_size = 256
        self.max_lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 0.3
        self.num_epochs = 100
        self.num_workers = 10

        self.device = None


class Trainer(object):

    def __init__(self, config: TrainerConfig):
        self.config = config

        self.model = config.model
        self.model.to(self.config.device)
        self.criterion = config.criterion
        if isinstance(self.criterion, nn.Module):
            self.criterion.to(self.config.device)

        self._init_dataloader()
        self._init_optimizer()

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            self.config.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True
        ) if self.config.train_dataset is not None else None

        self.test_loader = DataLoader(
            self.config.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        ) if self.config.test_dataset is not None else None

    def _init_optimizer(self):
        # Get optimizer constructor.
        Optimizer = getattr(optim, self.config.optimizer)

        # Create optimizer.
        opt_args = {
            'params': [*self.model.parameters()],
            'lr': self.config.max_lr,
            'weight_decay': self.config.weight_decay,
            'momentum': self.config.momentum,  # SGD, RMSprop
            'betas': (self.config.momentum, 0.999),  # Adam*
        }
        co = Optimizer.__init__.__code__
        self.optimizer: optim.optimizer = Optimizer(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })

        # Create scheduler.
        num_loops = self.config.num_epochs * len(self.train_loader)
        self.scheduler = LRScheduler(self.optimizer, CosineWarmupDecay(num_loops))

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.config.device)
        y = y.to(self.config.device)

        y_ = self.model(x)
        loss = self.criterion(y_, y)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), 0.1, math.inf)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.detach().cpu()

    def predict_step(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.config.device)
            y_ = self.model(x)
            y_ = y_.argmax(-1)
            return y_.detach().cpu()

    def train(self):
        if self.train_loader is None:
            return

        loss_g = None
        for epoch in range(self.config.num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                x, y = doc['image'], doc['label']
                loss = self.train_step(x, y)
                loss_g = 0.9 * loss_g + 0.1 * float(loss) if loss_g is not None else float(loss)
                lr = self.optimizer.param_groups[0]['lr']
                info = f'[{epoch + 1}/{self.config.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)

            metrics = self.evaluate()
            if metrics:
                print(f'[{epoch + 1}/{self.config.num_epochs}] L={loss_g:.06f}', end='')
                for k, v in metrics.items():
                    print(f' {k}={v:.04f}', end='')
                print()

    def evaluate(self):
        if self.test_loader is None:
            return

        meter = ClassificationMeter()
        self.model.eval()
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            x, y = doc['image'], doc['label']
            y_ = self.predict_step(x)
            meter.update(output=y_.numpy(), target=y.numpy())
        return {
            'Acc': meter.accuracy(),
            'F1': meter.f1().mean()
        }
