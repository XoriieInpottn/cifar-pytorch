#!/usr/bin/env python3


import os

import numpy as np
from docset import DocSet
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class Cifar10Dataset(Dataset):

    def __init__(self, path: str, train):
        super(Cifar10Dataset, self).__init__()
        self.docs = DocSet(path, 'r')
        self.aug = iaa.SomeOf((0, None), [
            iaa.Fliplr(),
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.SomeOf((1, None), [
                    iaa.WithChannels(0, iaa.Multiply((0.985, 1.015))),
                    iaa.WithChannels(1, iaa.Multiply((0.3, 1.7))),
                    iaa.WithChannels(2, iaa.Multiply((0.6, 1.4)))
                ])
            ),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 0.1)),
                iaa.Sharpen((0.0, 0.1))
            ]),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)
        ]) if train else iaa.Identity()

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        doc = self.docs[i]
        image = doc['feature']
        image = self.aug(image=image)
        label = doc['label']

        image = np.array(image, dtype=np.float32)
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        return {
            'image': image,
            'label': label
        }


def create_data_loader(path: str, batch_size: int, train: bool = False) -> DataLoader:
    dataset = (
        Cifar10Dataset(path, train)
        if isinstance(path, str) else
        ConcatDataset([Cifar10Dataset(_path, train) for _path in path])
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=max(4, os.cpu_count() // 8),
        pin_memory=True,
        persistent_workers=train
    )
    return data_loader
