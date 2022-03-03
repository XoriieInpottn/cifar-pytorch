#!/usr/bin/env python3


import argparse
import os
import random
import shutil

import cv2 as cv
import numpy as np
import torch

from model import create_model
from trainer import Trainer

cv.setNumThreads(0)

np.random.seed(0)
random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--backbone', required=True)

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.93)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=100)

    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--note')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.output_dir is not None:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    args.train_data_path = os.path.join(args.data_path, 'train.ds')
    args.test_data_path = os.path.join(args.data_path, 'test.ds')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model(args.backbone, args.num_classes)
    model = model.to(args.device)

    kwargs = {}
    co = Trainer.__init__.__code__
    for name in co.co_varnames[1:co.co_argcount]:
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    Trainer(model, **kwargs).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
