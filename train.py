#!/usr/bin/env python3

import argparse
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch import nn

from dataset import TrainDataset, TestDataset
from model import ModelConfig, Model
from trainer import TrainerConfig, Trainer

cv.setNumThreads(0)

np.random.seed(0)
random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--num-classes', type=int, required=True)

    parser.add_argument('--backbone', required=True)
    parser.add_argument('--non-lin')
    parser.add_argument('--norm')

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=10)

    parser.add_argument('--note')
    args = parser.parse_args()

    train_dataset = TrainDataset(os.path.join(args.data_path, 'train.ds'))
    test_dataset = TestDataset(os.path.join(args.data_path, 'test.ds'))

    model_config = ModelConfig()
    model_config.load(args)
    model = Model(model_config)
    print('==== Model config ====')
    print(model_config)

    trainer_config = TrainerConfig()
    trainer_config.load(args)
    trainer_config.model = model
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer_config.criterion = nn.CrossEntropyLoss()
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer = Trainer(trainer_config)
    print('==== Trainer config ====')
    print(trainer_config)

    trainer.train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
