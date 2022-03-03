#!/usr/bin/env python3


from docset import DocSet
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

import torchstocks


class Cifar10Dataset(Dataset):

    def __init__(self, path: str, train):
        super(Cifar10Dataset, self).__init__()
        self.docs = DocSet(path, 'r')
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Pad(4, keep_size=False),
            iaa.CropToFixedSize(32, 32),
        ]) if train else iaa.Identity()

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        doc = self.docs[i]
        image = doc['image']
        image = self.aug(image=image)
        image = torchstocks.utils.image.encode_image(image)
        label = doc['label']
        return {
            'image': image,
            'label': label
        }
